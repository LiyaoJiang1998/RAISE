from typing import Dict, List, cast, Optional, Tuple
from typing_extensions import Literal
import asyncio
from pathlib import Path
import os
import json
import copy

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field
import torch
from PIL import Image
from pydantic import BaseModel, Field


from graph.context import Context
from graph.state import InputState, State
from graph.utils import data_url_to_image, image_to_data_url, resize_to_max_resolution
from graph.models import ainvoke_with_timeout_retry
from graph.models_nvila import evaluate_nvila_score


class VerifierGenerationOutput(BaseModel):
    """Output Schema for the Verifier"""
    verifier_reasoning: str = Field(...,
        description="Let's think step by step. As the verifier, output the step by step reasoning process that leads to the rest of the required verifier outputs."
    )
    current_image_caption: str = Field(...,
        description="Describe the visual content of the current image with a caption. Strictly write what you see in the image, avoid any assumptions."
    )
    questions_answers_and_explanations: List[Tuple[str, Literal["Yes", "No"], str]] = Field(...,
        description="Base on looking at the current image visual content and current_image_caption, answer each question in the binary questions list with Yes (satisfied) or No (unsatisfied), and provide an explanation for each answer. Each item in this list is a tuple of (<question>, <Yes/No>, <explanation>)."
    )
    verifier_summary: str = Field(...,
        description="Summarize your verification result outputs to give suggestions to the analyzer for refining its next requirements analysis. Which requirements are satisfied? Which requirements are not satisfied?"
    )
    all_satisfied: bool = Field(...,
        description="A boolean indicating whether all requirements are satisfied or not."
    )


async def verifier(
    state: State, 
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> Command[Literal["analyzer", "__end__"]]:
    """
    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.
        runtime (Runtime): The runtime context containing model and system prompt information.

    Returns:
        dict: A dictionary containing the model's response message.
    """

    # Extract analyzer output from the state.analyzer_outputs
    if state.analyzer_outputs:
        analyzer_output = state.analyzer_outputs[-1]
        requirements_analysis_str = json.dumps(analyzer_output["requirements_analysis"])
        binary_questions_str = json.dumps(analyzer_output["binary_questions"])
    else:
        raise ValueError("No verifier input found (i.e. no analyzer output found in state.analyzer_outputs).")
    
    # Extract executor output from the state.executor_outputs
    if state.executor_outputs:
        executor_output_list = state.executor_outputs[-1]
    else:
        raise ValueError("No verifier input found (i.e. no executor output found in state.executor_outputs).")

    round_sample_scores = [executor_output_list[i]["score"] for i in range(len(executor_output_list))]
    if round_sample_scores[0] is not None:
        round_best_sample_idx, _ = max(enumerate(round_sample_scores), key=lambda x: (x[1], x[0]))  # prefer higher score, then later index
    else:
        round_best_sample_idx = None
    
    verifier_outputs = []
    verifier_image_data_urls = []
    for sample_idx, executor_output in enumerate(executor_output_list):
        input_image_data_url = executor_output["input_image_data_url"]
        output_image_data_url = executor_output["output_image_data_url"]

        if (round_best_sample_idx is None) or (sample_idx == round_best_sample_idx):
            image_size = executor_output["image_size"]
            if state.use_caption_and_grounding:
                detection_caption = executor_output["detection_caption"]
                detection_boxes = executor_output["detection_boxes"]
                detection_labels= executor_output["detection_labels"]
                detection_depths= executor_output["detection_depths"]
                
                detected_region_info_list = []
                for i, (box, label, depth) in enumerate(zip(detection_boxes, detection_labels, detection_depths)):
                    detected_region_info_list.append(f"Detected Region {i} - Region Label: '{label}', Bounding Box: {box}, Average Depth: {depth}")
                detected_region_info = "\n".join(detected_region_info_list)
                
            if input_image_data_url is not None:
                input_image = data_url_to_image(input_image_data_url)
                input_image = resize_to_max_resolution(input_image, max_resolution=state.vlm_max_resolution)
                input_image_data_url_vlm = image_to_data_url(input_image)
            else:
                input_image_data_url_vlm = None
                
            output_image = data_url_to_image(output_image_data_url)
            output_image = resize_to_max_resolution(output_image, max_resolution=state.vlm_max_resolution)
            output_image_data_url_vlm = image_to_data_url(output_image)

            # Initialize the system prompt, and output structure, prepare VLM Inputs
            output_class = VerifierGenerationOutput
            system_message = runtime.context.system_prompt_verifier_generation
            
            verifier_input_prompt = f"requirements_analysis: \n{requirements_analysis_str}"
            verifier_input_prompt += f"\n\nbinary_questions: \n{binary_questions_str}"
            if state.use_caption_and_grounding:
                verifier_input_prompt += f"\n\ncurrent_image_caption: {detection_caption}"
                verifier_input_prompt += f"\n\nimage_size: {image_size}"
                verifier_input_prompt += f"\n\ndetected_region_info: \n{detected_region_info}"
            else:
                verifier_input_prompt += f"\n\nimage_size: {image_size}"
                verifier_input_prompt += f"\n\ncurrent_image_caption, detected_region_info: not available. Obtain the current_image_caption by visually inspecting the provided current_image and summarizing its content."
            
            verifier_input = [{"role": "system", "content": system_message}, 
                            AIMessage(content=[{"type": "text", "text": "Here is the current_image to be verified:"},
                                                {"type": "image_url", "image_url": {"url": output_image_data_url_vlm}}]),
                            AIMessage(content=[{"type": "text", "text": verifier_input_prompt}]),]
                
            # Call the VLM model to get response
            verifier_empty_allowed_attempts = 20
            verifier_current_attempt = 0
            while verifier_current_attempt < verifier_empty_allowed_attempts:
                verifier_output = cast(VerifierGenerationOutput, await ainvoke_with_timeout_retry(state=state, runtime=runtime, inp=verifier_input, output_class=output_class))
                
                if verifier_output.verifier_reasoning == "" or verifier_output.verifier_summary == "" or len(verifier_output.questions_answers_and_explanations) == 0:
                    verifier_current_attempt += 1
                    print(f"verifier | Empty verifier output, retrying... {verifier_current_attempt}/{verifier_empty_allowed_attempts}")
                    verifier_input[0]["content"] += f"\n\nRetry number: {verifier_current_attempt}. Your previous response did not follow the required output format. Please make sure to provide a complete response following the required output format."
                else:
                    break
            else:
                raise ValueError(f"verifier | Empty verifier output after {verifier_empty_allowed_attempts} retries. Cannot proceed.")
        else:
            # For non-best samples, fill in dummy outputs
            verifier_output = VerifierGenerationOutput(
                verifier_reasoning="",
                current_image_caption="",
                questions_answers_and_explanations=[],
                verifier_summary="",
                all_satisfied=False
            )

        verifier_outputs.append(verifier_output)
        verifier_image_data_urls.append(output_image_data_url)
    
    # Pick the image instance with highest score in the current round
    verifier_output_dicts = [verifier_output.model_dump() for verifier_output in verifier_outputs]
    no_count_list = []
    question_count_list = []
    score_list = []
    for executor_output, verifier_output_dict in zip(executor_output_list, verifier_output_dicts):
        no_count, question_count = 0, 0
        for question_index, (question, answer, explanation) in enumerate(verifier_output_dict["questions_answers_and_explanations"]):
            question_count += 1
            if answer == "No":
                no_count += 1
        
        score = executor_output["score"]

        no_count_list.append(no_count)
        question_count_list.append(question_count)
        score_list.append(score)

        verifier_output_dict["no_count"] = no_count
        verifier_output_dict["question_count"] = question_count
        verifier_output_dict["score"] = score
        verifier_output_dict["is_best"] = False
        verifier_output_dict["original_prompt"] = executor_output["original_prompt"]
        verifier_output_dict["current_prompt"] = executor_output["executed_prompt"]

    # Select the best verifier output with the highest score
    best_idx, _ = max(enumerate(score_list), key=lambda x: (x[1], x[0]))  # prefer higher score, then later index
    
    best_score = score_list[best_idx]
    verifier_output_dicts[best_idx]["is_best"] = True
    
    # If Saving json
    if state.saving_files:
        await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
        await asyncio.to_thread(Path(os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"verifier_{state.current_round}.json")).write_text, json.dumps(verifier_output_dicts, indent=4), encoding="utf-8")
    
    # Prepare command update dict
    verifier_output = verifier_outputs[best_idx]

    verifier_output_string = f"Original Prompt: {verifier_output_dicts[best_idx]['original_prompt']}\n\nCurrent Prompt:\n{verifier_output_dicts[best_idx]['current_prompt']}\n" + \
                             f"\nVerifier Score: {best_score}\n\nVerifier Reasoning:\n{verifier_output.verifier_reasoning}\n" + \
                             f"\nVerifier Output:\n```json\n{verifier_output.model_dump_json(indent=4, exclude={'verifier_reasoning'})}\n```"
    
    command_update_dict = {"messages": [AIMessage(content=verifier_output_string)],
                            "verifier_outputs": [verifier_output_dicts],
                            "current_round": state.current_round + 1}
    
    # Determine the best_round and best_round_sample across all rounds so far
    temp_verifier_outputs = state.verifier_outputs + [verifier_output_dicts]
    # Obtain image from the round with highest score (break tie by prefering the later round, later sample)
    # First set to return last round last output
    best_score = temp_verifier_outputs[-1][-1]["score"]
    best_verifier_image_data_url = state.executor_outputs[-1][-1]["output_image_data_url"]
    best_round, best_round_sample = state.current_round, len(temp_verifier_outputs[-1])
    # Go over all rounds to find the best
    for i, round_output_dicts in enumerate(temp_verifier_outputs):
        for j, round_output_dict in enumerate(round_output_dicts):
            if round_output_dict["score"] > best_score:
                best_score = round_output_dict["score"]
                best_verifier_image_data_url = state.executor_outputs[i][j]["output_image_data_url"]
                best_round, best_round_sample = i + 1, j + 1

    # Update the best_round and best_round_sample in the state
    command_update_dict["best_round"] = best_round
    command_update_dict["best_round_sample"] = best_round_sample
    
    command_update_dict["round_best_images"] = [best_verifier_image_data_url]
    command_update_dict["max_rounds"] = state.max_rounds
    
    best_all_satisfied = temp_verifier_outputs[best_round-1][best_round_sample-1]["all_satisfied"]
    # If not max rounds, go to next round
    if state.current_round < state.max_rounds:
        if best_all_satisfied and state.current_round >= state.min_rounds:
            goto = "__end__"
        else:
            goto = "analyzer"
    else:
        goto = "__end__"

    # If at the end, save the final output image
    if goto == "__end__":    
        # Return the best output image
        command_update_dict["output_images"] = [best_verifier_image_data_url]
        
        if state.saving_files:
            generated_image = data_url_to_image(best_verifier_image_data_url)
            await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
            await asyncio.to_thread(generated_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_selected.jpg"))
        
    command = Command(
        update=command_update_dict,
        goto=goto
    )
        
    return command

