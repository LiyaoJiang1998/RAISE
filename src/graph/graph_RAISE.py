import argparse
import asyncio
import os
from pathlib import Path
from langgraph.graph import StateGraph
from tqdm import tqdm

from graph.context import Context
from graph.state import InputState, State
from graph.utils import save_result, data_url_to_image
from graph.node_analyzer import analyzer
from graph.node_rewriter import rewriter
from graph.node_executor import executor
from graph.node_verifier import verifier

# Graph for Image Generation
builder = StateGraph(State, input_schema=State, context_schema=Context)

# Nodes
builder.add_node("analyzer", analyzer) # Analyzer Node
builder.add_node("rewriter", rewriter) # Rewriter Node
builder.add_node("executor", executor) # Executor Node
builder.add_node("verifier", verifier) # Verifier Node

# Edges
builder.add_edge("__start__", "analyzer") # Set the entrypoint as `analyzer`

# Compile the builder into an executable graph
graph_RAISE = builder.compile(name="Image Generation - RAISE Framework")


async def test_on_generation_benchmark_txt(benchmark_txt_path, 
                                           saving_path,
                                           saving_files=False,
                                           seed=42, 
                                           base_url="127.0.0.1:11434"):
    # Read all prompts (strip empty lines)
    with Path(benchmark_txt_path).open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    await asyncio.to_thread(os.makedirs, saving_path, exist_ok=True)
    
    # Loop sequentially with tqdm
    for i, prompt in enumerate(tqdm(prompts, desc="Processing Prompts")):
        if os.path.exists(os.path.join(saving_path, f"{i:06d}_0_result.json")):
            continue
        
        for retry_i in range(5):
            try:
                image_seed = seed
                llm_seed = seed + retry_i*100
                print(f"\nPrompt: {prompt}")
                
                result = await graph_RAISE.ainvoke(
                    {"original_prompt": prompt,
                    "image_seed": image_seed,
                    "save_original_prompt_image": True,
                    "saving_files": saving_files,
                    "saving_path": saving_path},
                    context=Context(seed=llm_seed, base_url=base_url),
                    config={"recursion_limit": 1000}
                )

                # Save the results
                current_path_prefix = f"{saving_path}/{i:06d}"
                save_result(result, current_path_prefix)
                # print(f"Result: {result}")
                break
            except Exception as e:
                # Retry on any exception from the LLM
                print(f"Error occurred: {e}")
                continue


async def async_generate_with_prompt(prompt,
                                     seed=42,
                                     save_idx=0, 
                                     saving_path="outputs/default/",
                                     saving_files=False, 
                                     base_url="127.0.0.1:11434"):
    os.makedirs(saving_path, exist_ok=True)
    
    result = await graph_RAISE.ainvoke(
        {"original_prompt": prompt,
         "image_seed": seed,
         "save_original_prompt_image": False,
         "saving_files": saving_files,
         "saving_path": saving_path},
        context=Context(seed=seed, base_url=base_url),
        config={"recursion_limit": 1000}
    )
    
    # obtain output image
    output_images = result["output_images"]
    output_image_data_url = output_images[-1]
    output_image = data_url_to_image(output_image_data_url)

    # Save the results
    current_path_prefix = f"{saving_path}/{save_idx:06d}"
    save_result(result, current_path_prefix)
        
    return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', default='127.0.0.1:11434', type=str, help='Host address for the ollama server')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for image generation')
    parser.add_argument('--prompts', default=None, type=str, nargs='+', help='List of prompts to generate images for')
    
    parser.add_argument('--benchmark_txt_path', default=None, type=str, help='Path to the benchmark txt file with prompts')
    parser.add_argument('--saving_path', default="outputs/default/", type=str, help='Path to save the results')
    parser.add_argument('--saving_files', type=bool, default=True, help='Whether to save the detailed generated files for developement/debugging')
    
    args = parser.parse_args()

    if args.prompts is not None:
        with asyncio.Runner() as runner:
            for idx, prompt in enumerate(tqdm(args.prompts, desc="Processing Prompts")):
                print(f"\nPrompt: {prompt}")
                result_image = runner.run(async_generate_with_prompt(
                                            prompt=prompt, 
                                            seed=args.seed,
                                            save_idx=idx,
                                            saving_path=args.saving_path,
                                            saving_files=args.saving_files, 
                                            base_url=args.base_url))
    
    elif args.benchmark_txt_path is not None:
        asyncio.run(test_on_generation_benchmark_txt(benchmark_txt_path=args.benchmark_txt_path, 
                                                    saving_path=args.saving_path,
                                                    saving_files=args.saving_files,
                                                    seed=args.seed,
                                                    base_url=args.base_url))
    
    else:
        print("Please provide either --prompts or --benchmark_txt_path.")

