from typing import Dict, List, cast
from typing_extensions import Literal
import asyncio
import json
import os
from pathlib import Path
import random

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
import torch
from PIL import Image
import numpy as np

from graph.context import Context
from graph.state import InputState, State
from graph.utils import data_url_to_image, image_to_data_url, resize_to_max_resolution
from graph.models_aux import get_aux_models, run_caption_ground_depth, run_referring_expression_segmentation
from graph.models_nvila import evaluate_nvila_score

MAX_SEED = np.iinfo(np.int32).max

# - Quality Model
def import_and_execute_quality_pipe(seed_offset, state, prompt):
    random.seed(state.image_seed + seed_offset)
    seed = random.randint(0, MAX_SEED)

    from graph.models_dm import get_quality_pipe
    generated_image = get_quality_pipe()(
        prompt=prompt,
        height=state.gen_height,
        width=state.gen_width,
        guidance_scale=state.gen_guidance_scale,
        num_inference_steps=state.gen_num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
    return generated_image, seed

# - Speed Model
def import_and_execute_speed_pipe(seed_offset, state, prompt):
    random.seed(state.image_seed + seed_offset)
    seed = random.randint(0, MAX_SEED)
    
    from graph.models_dm import get_speed_pipe
    generated_image = get_speed_pipe()(
        prompt=prompt,
        height=state.gen_height,
        width=state.gen_width,
        guidance_scale=state.gen_guidance_scale_speed,
        num_inference_steps=state.gen_num_inference_steps_speed,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
    return generated_image, seed

# - Edit Model
def import_and_execute_edit_pipe(seed_offset, state, prompt, input_image):
    random.seed(state.image_seed + seed_offset)
    seed = random.randint(0, MAX_SEED)
    
    from graph.models_dm import get_edit_pipe
    width, height = input_image.size
    edited_image = get_edit_pipe()(
        image=input_image,
        prompt=prompt,
        guidance_scale=state.edit_guidance_scale,
        width=width,
        height=height,
        num_inference_steps=state.edit_num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    return edited_image, seed



async def executor(
    state: State, 
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> Command[Literal["verifier", "__end__"]]:
    """Executor node that processes the rewriter output and generates or edits the image

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.
        runtime (Runtime): The runtime context containing model and system prompt information.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    
    if state.use_caption_and_grounding:
        SAM2_MODEL, SAM2_PREDICTOR, FLORENCE2_MODEL, FLORENCE2_PROCESSOR, MIDAS_DEPTH_ESTIMATOR = get_aux_models()
        
    original_prompt_image_data_url = None
    executor_output_strings = []
    executor_output_list = []
    
    if not state.rewriter_outputs:
        raise ValueError("No rewriter_output found in state.rewriter_outputs")
    
    if state.rewriter_outputs[-1]["editing"]:
        num_planned_edits = len(state.rewriter_outputs[-1]["editing"]["planned_edits"])
        random.seed(state.image_seed + state.current_round - 1) # for reproducibility
        random_idx = 0 if num_planned_edits == 1 else random.randint(1, max(1, num_planned_edits-1))
        if num_planned_edits == 0:
            random_edit_prompt = state.rewriter_outputs[-1]["editing"]["comprehensive_editing_prompt"]
        else:
            random_edit_prompt = state.rewriter_outputs[-1]["editing"]["planned_edits"][random_idx]
    
    if state.current_round == 1:
        executor_actions_list = [
            ["generation_original_prompt", state.original_prompt],                              # Generation with original prompt
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
        ]
    elif state.current_round <= state.min_rounds:
        executor_actions_list = [
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.original_prompt],                                              # Generation with original prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
        ]
    else:
        executor_actions_list = [
            ["editing", random_edit_prompt],                                                    # Edit with rewriter's random planned edits (except top-1)
            ["editing", state.rewriter_outputs[-1]["editing"]["single_editing_prompt"]],        # Edit with rewriter's top-1 planned edits
            ["editing", state.rewriter_outputs[-1]["editing"]["comprehensive_editing_prompt"]], # Edit with rewriter's comprehensive editing prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
            ["generation", state.rewriter_outputs[-1]["generation"]["adjusted_prompt"]],        # Generation with rewriter's adjusted prompt
        ]
    
    for current_sample, (model_choice, prompt) in enumerate(executor_actions_list):
        current_sample = current_sample + 1 # To make it 1-indexed for better readability
        seed_offset = sum([len(temp_output) for temp_output in state.executor_outputs]) + (current_sample-1)
        
        if state.verifier_outputs:
            # Extract current image from the state.executor_outputs
            current_image_data_url = state.executor_outputs[state.best_round-1][state.best_round_sample-1]["output_image_data_url"]
            current_image = data_url_to_image(current_image_data_url)
            current_image = resize_to_max_resolution(current_image, max_resolution=state.edit_max_resolution)
            current_image_data_url = image_to_data_url(current_image)
        else:
            current_image_data_url = None
        
        # Execute the appropriate model based on model_choice:
        print(f"executor | round: {state.current_round} | sample: {current_sample} | model_choice: {model_choice}")
        # - Editing Model
        if model_choice == "editing":
            image_to_edit = current_image
            edited_image, seed_used = await asyncio.to_thread(import_and_execute_edit_pipe, seed_offset, state, prompt, image_to_edit) # Execute in new thread to make async
            # If Saving Files
            if state.saving_files:
                await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
                await asyncio.to_thread(edited_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_{state.current_round}_{current_sample}.jpg"))

            return_input_image_data_url = image_to_data_url(image_to_edit)
            output_image = edited_image
            output_image_data_url = image_to_data_url(output_image)
            executor_output_strings.append(f"Executor: Edited Image with Adjusted Prompt")
            
        # - Generation Model
        elif model_choice == "generation":
            generated_image, seed_used = await asyncio.to_thread(import_and_execute_quality_pipe, seed_offset, state, prompt) # Execute in new thread to make async

            # If Saving Files
            if state.saving_files:
                await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
                await asyncio.to_thread(generated_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_{state.current_round}_{current_sample}.jpg"))
                
            return_input_image_data_url = None
            output_image = generated_image
            output_image_data_url = image_to_data_url(output_image)
            executor_output_strings.append(f"Executor: Generated Image with Adjusted Prompt")
        
        # - Generation Model with Original Prompt
        elif model_choice == "generation_original_prompt":
            generated_image, seed_used = await asyncio.to_thread(import_and_execute_quality_pipe, seed_offset, state, prompt) # Execute in new thread to make async

            if state.save_original_prompt_image:
                original_prompt_image_data_url = image_to_data_url(generated_image)
                
            # If Saving Files
            if state.saving_files:
                await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
                await asyncio.to_thread(generated_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_{state.current_round}_{current_sample}.jpg"))
                
            return_input_image_data_url = None
            output_image = generated_image
            output_image_data_url = image_to_data_url(output_image)
            executor_output_strings.append(f"Executor: Generated Image with Original Prompt")
        
        else:
            raise ValueError(f"Invalid model_choice '{model_choice}' in executor.")
        
        # Use Aux Models to extract caption and bounding boxes
        if state.use_caption_and_grounding:
            caption_text, integer_boxes, labels, avg_depths, annotated_image, depth_image = run_caption_ground_depth(
                image=output_image,
                florence2_model=FLORENCE2_MODEL, florence2_processor=FLORENCE2_PROCESSOR,
                sam2_model=SAM2_MODEL, sam2_predictor=SAM2_PREDICTOR,
                midas_depth_estimator=MIDAS_DEPTH_ESTIMATOR,
                caption_type="more_detailed_caption",
            )
            if state.saving_files:
                await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
                await asyncio.to_thread(annotated_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_{state.current_round}_{current_sample}_annotated.jpg"))
                await asyncio.to_thread(depth_image.save, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"output_{state.current_round}_{current_sample}_depth.jpg"))
        else:
            caption_text, integer_boxes, labels, avg_depths = None, None, None, None
            
        score = evaluate_nvila_score(image=output_image, 
                                        prompt=state.original_prompt, 
                                        seed=runtime.context.seed)
        
        # Prepare executor output
        executor_output_list.append({
            "model_choice": model_choice,
            "seed_used": seed_used,
            "input_image_data_url": return_input_image_data_url,
            "output_image_data_url": output_image_data_url,
            "original_prompt": state.original_prompt,
            "executed_prompt": prompt,
            "image_size": edited_image.size if model_choice == "editing" else generated_image.size,
            "detection_caption": caption_text,
            "detection_boxes": integer_boxes,
            "detection_labels": labels,
            "detection_depths": avg_depths,
            "score": score,
        })
        
    # If Saving Files
    if state.saving_files:
        executor_output_list_exclude = [{key: value for key, value in executor_output_dict.items() if key not in ["input_image_data_url", "output_image_data_url"]}  for  executor_output_dict in executor_output_list]# Exclude image data URLs from saved JSON
        executor_output_json = json.dumps(executor_output_list_exclude, indent=4)
        await asyncio.to_thread(os.makedirs, os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64]), exist_ok=True)
        await asyncio.to_thread(Path(os.path.join(state.saving_path, state.original_prompt.replace(' ','-')[:64], f"executor_{state.current_round}.json")).write_text, executor_output_json, encoding="utf-8")

    update_dict = {"messages": [AIMessage(content=message_content) for message_content in executor_output_strings],
                    "executor_outputs": [executor_output_list]}
    if original_prompt_image_data_url:
        update_dict["original_prompt_image"] = original_prompt_image_data_url
    
    goto = "__end__" if state.agent_setup == 'original' else "verifier"
    
    command = Command(
        update=update_dict,
        goto=goto
    )
    
    return command

