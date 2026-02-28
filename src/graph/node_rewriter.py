from typing import Dict, List, cast, Optional
from typing_extensions import Literal
import asyncio
from pathlib import Path
import os
import json

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


class RewriterEditingOutput(BaseModel):
    """Output Schema for the Rewriter."""
    rewriter_reasoning: str = Field(...,
        description="Let's think step by step. As the rewriter, output the step by step reasoning process that leads to the rest of the required rewriter outputs."
    )
    original_prompt: str = Field(...,
        description="From analyzer_output, the original prompt."
    )
    current_prompt: str = Field(...,
        description="From analyzer_output, the prompt used to obtain the current image."
    )
    planned_edits: List[str] = Field(...,
        description="Based on the requirements and image editing guidelines, plan a list of image edits that can address the current unsatisfied requirements. Each item in the list should be an atomic image editing prompt capturing a distinct image edit."
    )
    single_editing_prompt: str = Field(...,
        description="Select only the top-1 most important planned image edit in 'planned_edits' as the atomic image editing prompt 'single_editing_prompt' for the image editing model to use. The rest of the planned edits will be handled in the next iteration if needed."
    )
    comprehensive_editing_prompt: str = Field(...,
        description="Combine all items from 'planned_edits' into a single, cohesive, natural-language image editing prompt 'comprehensive_editing_prompt' that captures every planned change for execution in one pass by the image editing model."
    )
    

class RewriterGenerationOutput(BaseModel):
    """Output Schema for the Rewriter."""
    rewriter_reasoning: str = Field(...,
        description="Let's think step by step. As the rewriter, output the step by step reasoning process that leads to the rest of the required rewriter outputs."
    )
    original_prompt: str = Field(...,
        description="From analyzer_output, the original prompt."
    )
    current_prompt: str = Field(...,
        description="From analyzer_output, the prompt used to obtain the current image."
    )
    planned_adjustments: List[str] = Field(...,
        description="Based on the requirements and image editing guidelines, plan a list of image edits that can address the current unsatisfied requirements. Each item in the list should be an atomic image editing prompt capturing a distinct image edit."
    )
    adjusted_prompt: str = Field(...,
        description="Apply the planned adjustments to the current prompt, and as a result get this adjusted prompt. If no adjustments proposed or needed, this adjusted prompt field should be same as current_prompt."
    )


async def rewriter(
    state: State, 
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> Command[Literal["executor"]]:
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
        # print(f"rewriter input original_prompt: \n{original_prompt}\n")
    else:
        raise ValueError("No original_prompt found in the rewriter input state.")
    
    # Extract analyzer output from the state.analyzer_outputs
    if state.analyzer_outputs:
        keys_to_include = ["analyzer_reasoning", "current_prompt", "satisfied_requirements", "unsatisfied_requirements"]
        analyzer_output_str = str({key: value for key, value in state.analyzer_outputs[-1].items() if key in keys_to_include})
    else:
        raise ValueError("No analyzer_output found in state.analyzer_outputs.")
        
    if state.verifier_outputs:
        # Extract current image from the state.executor_outputs
        current_image_data_url = state.executor_outputs[state.best_round-1][state.best_round_sample-1]["output_image_data_url"]
        current_image = data_url_to_image(current_image_data_url)
        current_image = resize_to_max_resolution(current_image, max_resolution=state.vlm_max_resolution)
        current_image_data_url = image_to_data_url(current_image)
    else:
        current_image_data_url = None
    
    round_rewriter_output_dict = {"generation": None, "editing": None}
    round_rewriter_output_strings = []
    
    if state.current_round > state.min_rounds:
        rewriter_tasks = ["generation", "editing"]
    else:
        rewriter_tasks = ["generation"]

    for rewriter_task in rewriter_tasks:
        # Initialize the system prompt, and output structure
        if rewriter_task == "editing": # Rewriter for Editing
            output_class = RewriterEditingOutput
            system_message = runtime.context.system_prompt_rewriter_editing
        else: # Rewriter for Generation
            output_class = RewriterGenerationOutput
            system_message = runtime.context.system_prompt_rewriter_generation    

        # Prepare VLM Input
        rewriter_input_prompt = f"original_prompt: \n{original_prompt}"
        rewriter_input_prompt += f"\n\nanalyzer_output: \n{analyzer_output_str}"
            
        if current_image_data_url:
            # Have current image
            rewriter_input = [{"role": "system", "content": system_message}, 
                                AIMessage(content=[{"type": "text", "text": "Here is the current_image:"},
                                                {"type": "image_url", "image_url": {"url": current_image_data_url}}]),
                                AIMessage(content=[{"type": "text", "text": rewriter_input_prompt}])]
        
        else:
            # Without image
            rewriter_input = [{"role": "system", "content": system_message}, 
                                AIMessage(content=[{"type": "text", "text": rewriter_input_prompt}])]
            
        # Call the VLM model to get response
        if rewriter_task == "editing":
            rewriter_output = cast(RewriterEditingOutput, await ainvoke_with_timeout_retry(state=state, runtime=runtime, inp=rewriter_input, output_class=output_class))
        else:
            rewriter_output = cast(RewriterGenerationOutput, await ainvoke_with_timeout_retry(state=state, runtime=runtime, inp=rewriter_input, output_class=output_class))
        round_rewriter_output_dict[rewriter_task] = rewriter_output.model_dump()
        
        rewriter_output_string = f"Rewriter Reasoning - {rewriter_task}:\n{rewriter_output.rewriter_reasoning}\n" + \
                                f"\nRewriter Output - {rewriter_task}:\n```json\n{rewriter_output.model_dump_json(indent=4, exclude={'rewriter_reasoning'})}\n```"
        round_rewriter_output_strings.append(rewriter_output_string)
        
    # If Saving Files
    if state.saving_files:
        await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
        await asyncio.to_thread(Path(os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"rewriter_{state.current_round}.json")).write_text, json.dumps(round_rewriter_output_dict, indent=4), encoding="utf-8")
            
    command_update_dict = {"messages": [AIMessage(content=message_content) for message_content in round_rewriter_output_strings],
                           "rewriter_outputs": [round_rewriter_output_dict]}
    
    command = Command(
        update=command_update_dict,
        goto="executor"
    )
    
    return command

