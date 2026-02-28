from typing import Dict, List, cast, Optional
from typing_extensions import Literal
import asyncio
from pathlib import Path
import os

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field
import torch
from PIL import Image

from graph.context import Context
from graph.state import InputState, State
from graph.utils import data_url_to_image, image_to_data_url, resize_to_max_resolution
from graph.models import ainvoke_with_timeout_retry



class AnalyzerGenerationOutput(BaseModel):
    """Output Schema for the Analyzer."""
    analyzer_reasoning: str = Field(...,
        description="Let's think step by step. As the analyzer, output the step by step reasoning process that leads to the rest of the required analyzer outputs."
    )
    original_prompt: str = Field(...,
        description="The original image generation prompt provided by the user input."
    )
    current_prompt: str = Field(...,
        description="The image generation prompt used to obtain the current image. If initial round, this should be same as the original_prompt."
    )
    requirements_analysis: List[str] = Field(...,
        description="List the requirements either explicitly or implicitly conveyed by the original_prompt, current_image, current_verifier_output, and reference_verifier_output. Each item in the list should be a sentence capturing a distinct requirement."
    )
    satisfied_requirements: List[str] = Field(...,
        description="Analyze the requirements_analysis list, original_prompt, current_image, and current_verifier_output: list the requirements that are already successfully achieved. If initial round and no current_verifier_output yet, this list should be empty. Do not judge a requirement as satisfied or not satisfied based on reference_verifier_output."
    )
    unsatisfied_requirements: List[str] = Field(...,
        description="Analyze the requirements_analysis list, original_prompt, current_image, and current_verifier_output: list the requirements that are not achieved and need to be addressed. If initial round and no current_verifier_output yet, this list should be same as the requirements_analysis list. Do not judge a requirement as satisfied or not satisfied based on reference_verifier_output."
    )
    binary_questions: List[str] = Field(...,
        description="Convert each requirement in the requirements analysis list to a binary question that can be clearly answered with Yes or No. This list should contain a list of binary verifiable questions corresponding to each of the requirements."
    )
    model_choice: Literal["continue", "ending"] = Field(...,
        description="Use 'ending' when the remaining very few unsatisfied requirements are not explicitly required by the original prompt and involve only minor aspects such as lighting, mood/atmosphere, camera aperture or depth of field, camera angle or perspective, lens or focal length, or composition/framing (e.g., rule-of-thirds). Otherwise default to use 'continue'."
    )


async def analyzer(
    state: State, 
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> Command[Literal["rewriter", "__end__"]]:
    """
    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.
        runtime (Runtime): The runtime context containing model and system prompt information.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Extract original prompt from state
    if state.original_prompt:
        original_prompt = state.original_prompt
        # print(f"analyzer input original_prompt: \n{original_prompt}\n")
    else:
        raise ValueError("No original_prompt found in the analyzer input state.")
    
    if state.verifier_outputs:
        # Extract current image from the state.executor_outputs
        current_image_data_url = state.executor_outputs[state.best_round-1][state.best_round_sample-1]["output_image_data_url"]
        current_image = data_url_to_image(current_image_data_url)
        current_image = resize_to_max_resolution(current_image, max_resolution=state.vlm_max_resolution)
        current_image_data_url = image_to_data_url(current_image)
        
        # Extract verifier output from the state.verifier_outputs
        keys_to_include = ["current_prompt", "questions_answers_and_explanations", "verifier_summary"]
        current_verifier_output_str = str({key: value for key, value in state.verifier_outputs[state.best_round-1][state.best_round_sample-1].items() if key in keys_to_include})
        verifier_output_str = f"current_verifier_output: \n{current_verifier_output_str}"
        
        if state.best_round != state.current_round-1:
            # If best round is not the previous round, also include the previous round verifier output
            reference_verifier_output_str = str({"reference_"+key: value for key, value in state.verifier_outputs[state.current_round-2][0].items() if key in keys_to_include})
        else:
            reference_verifier_output_str = "Same as current_verifier_output."
        verifier_output_str += f"\n\nreference_verifier_output: \n{reference_verifier_output_str}"    
    else:
        current_image_data_url = None
        verifier_output_str = "Initial Round, don't have current_verifier_output and reference_verifier_output yet."
    # print(f"analyzer input from verifier_output: \n{verifier_output_str}\n")
    
    # Initialize the system prompt, and output structure
    output_class = AnalyzerGenerationOutput
    system_message = runtime.context.system_prompt_analyzer_generation

    # Prepare VLM Input
    analyzer_input_prompt = f"original_prompt: \n{original_prompt}"
    analyzer_input_prompt += f"\n\n{verifier_output_str}"
    
    if current_image_data_url:
        # Have current image
        analyzer_input = [{"role": "system", "content": system_message}, 
                            AIMessage(content=[{"type": "text", "text": "Here is the current_image:"},
                                               {"type": "image_url", "image_url": {"url": current_image_data_url}}]),
                            AIMessage(content=[{"type": "text", "text": analyzer_input_prompt}])]
    else:
        # Without image
        analyzer_input = [{"role": "system", "content": system_message}, 
                            AIMessage(content=[{"type": "text", "text": analyzer_input_prompt}])]

    # Call the VLM model to get response
    analyzer_output = cast(AnalyzerGenerationOutput, await ainvoke_with_timeout_retry(state=state, runtime=runtime, inp=analyzer_input, output_class=output_class))
        
    if state.current_round <= state.min_rounds:
        analyzer_output.model_choice = "continue" # Initial round has to be continue, no execution yet

    # print(f"analyzer | analyzer_output: \n{analyzer_output.model_dump_json(indent=4)}\n")
    # If Saving Files
    if state.saving_files:
        await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
        await asyncio.to_thread(Path(os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"analyzer_{state.current_round}.json")).write_text, analyzer_output.model_dump_json(indent=4), encoding="utf-8")

    analyzer_output_string = f"Analyzer Reasoning:\n{analyzer_output.analyzer_reasoning}\n" + \
                            f"\nAnalyzer Output:\n```json\n{analyzer_output.model_dump_json(indent=4, exclude={'analyzer_reasoning'})}\n```"
    
    command_update_dict = {"messages": [AIMessage(content=analyzer_output_string)],
                           "analyzer_outputs": [analyzer_output.model_dump()]}
    
    if analyzer_output.model_choice == "ending" and state.current_round > state.min_rounds:
        goto = "__end__"
    else:
        goto = "rewriter"
        
    # If at the end, save the final output image            
    if goto == "__end__":    
        # Obtain the image corresponding to the best_round and best_round_sample
        best_verifier_image_data_url = state.executor_outputs[state.best_round-1][state.best_round_sample-1]["output_image_data_url"]
        # Return the output image
        command_update_dict["output_images"] = [best_verifier_image_data_url]
        
        if state.saving_files:
            generated_image = data_url_to_image(best_verifier_image_data_url)
            await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
            await asyncio.to_thread(generated_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_selected.jpg"))
    
    command = Command(
        update=command_update_dict,
        goto=goto,
    )

    return command

