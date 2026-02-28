import argparse
import asyncio
import os
from pathlib import Path
from langgraph.graph import StateGraph
from tqdm import tqdm

from graph.context import Context
from graph.state import InputState, State
from graph.utils import data_url_to_image
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
builder.add_edge("__start__", "executor") # Set the entrypoint as `executor`
builder.add_edge("executor", "__end__")

# Compile the builder into an executable graph
graph_original = builder.compile(name="Image Generation - With Original Prompt")


async def test_on_generation_benchmark_txt(benchmark_txt_path, 
                                           saving_path,
                                           saving_files=False,
                                           seed=42,
                                           base_url="127.0.0.1:11434",
):
    # Read all prompts (strip empty lines)
    with Path(benchmark_txt_path).open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    await asyncio.to_thread(os.makedirs, saving_path, exist_ok=True)

    # Loop sequentially with tqdm
    for save_idx, prompt in enumerate(tqdm(prompts, desc="Processing Prompts")):
        if os.path.exists(os.path.join(saving_path, f"{save_idx:06d}_0_result.json")):
            continue
        print(f"\nPrompt: {prompt}")

        result = await graph_original.ainvoke(
            input={"original_prompt": prompt,
                   "image_seed": seed,
                    "rewriter_outputs": [{"generation":{"adjusted_prompt": prompt}, "editing": None}],
                    "analyzer_outputs": [{"model_choice": "continue"}],
                    "agent_setup": "original",
                    "saving_files": saving_files,
                    "saving_path": saving_path},
            context=Context(seed=seed, base_url=base_url),
            config={"recursion_limit": 1000}
        )
        # You can print or save result here if needed
        # print(f"Result: {result}")
        
        # obtain output image
        executor_output = result["executor_outputs"][-1][-1]
        output_image_data_url = executor_output["output_image_data_url"]
        output_image = data_url_to_image(output_image_data_url)
        # Save the results
        current_path_prefix = f"{saving_path}/{save_idx:06d}"
        output_image.save(f"{current_path_prefix}_0_original_prompt.jpg")
            

async def async_generate_with_original_prompt(prompt, 
                                              seed=42, 
                                              save_idx=0, 
                                              saving_files=False, 
                                              saving_path="outputs/default_original_prompt/",
                                              base_url="127.0.0.1:11434"):

    os.makedirs(saving_path, exist_ok=True)

    result = await graph_original.ainvoke(
        {"original_prompt": prompt,
         "image_seed": seed,
         "rewriter_outputs": [{"generation":{"adjusted_prompt": prompt}, "editing": None}],
         "analyzer_outputs": [{"model_choice": "continue"}],
         "agent_setup": "original",         
         "saving_files": saving_files,
         "saving_path": saving_path},
        context=Context(seed=seed, base_url=base_url),
        config={"recursion_limit": 1000}
    )
    
    # obtain output image
    executor_output = result["executor_outputs"][-1][-1]
    output_image_data_url = executor_output["output_image_data_url"]
    output_image = data_url_to_image(output_image_data_url)
    # Save the results
    current_path_prefix = f"{saving_path}/{save_idx:06d}"
    output_image.save(f"{current_path_prefix}_0_original_prompt.jpg")
        
    return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', default='127.0.0.1:11434', type=str, help='Host address for the ollama server')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--prompts', default=None, type=str, nargs='+', help='List of prompts to generate images for')
    
    parser.add_argument('--benchmark_txt_path', default=None, type=str, help='Path to the benchmark txt file with prompts')
    parser.add_argument('--saving_path', default="outputs/default_original_prompt/", type=str, help='Path to save the results')
    parser.add_argument('--saving_files', type=bool, default=True, help='Whether to save the detailed generated files for developement/debugging')
    
    args = parser.parse_args()

    if args.prompts is not None:
        with asyncio.Runner() as runner:
            for idx, prompt in enumerate(tqdm(args.prompts, desc="Processing Prompts")):
                print(f"\nPrompt: {prompt}")
                result_image = runner.run(async_generate_with_original_prompt(
                                            prompt=prompt, 
                                            seed=args.seed,
                                            save_idx=idx,
                                            saving_files=args.saving_files,
                                            saving_path=args.saving_path,
                                            base_url=args.base_url))
    
    elif args.benchmark_txt_path is not None:
        asyncio.run(test_on_generation_benchmark_txt(
                        benchmark_txt_path=args.benchmark_txt_path, 
                        saving_path=args.saving_path,
                        saving_files=args.saving_files,
                        seed=args.seed,
                        base_url=args.base_url))
    
    else:
        print("Please provide either --prompts or --benchmark_txt_path.")

