"""Adapted from GenEval benchmark: https://github.com/djghosh13/geneval/tree/main"""

import argparse
import json
import os
from tqdm import tqdm
import asyncio

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything

from src.graph.graph_RAISE import async_generate_with_prompt
from src.graph.graph_original import async_generate_with_original_prompt


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_original_prompt",
        action="store_true",
        help="whether to use the original prompt or RAISE framework",
        default=False,
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt",
        default="benchmarks/geneval_evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/geneval_results/",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="number of samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="127.0.0.1:11434",
        help="Base URL for the ollama API server"
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    
    with asyncio.Runner() as runner:
        
        # Load prompts
        with open(opt.metadata_file) as fp:
            metadatas = [json.loads(line) for line in fp]
            
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert opt.batch_size == 1

        for index, metadata in tqdm(enumerate(metadatas), total=len(metadatas), desc="Prompts Progress"):
            if os.path.exists(os.path.join(opt.outdir, "agent_results", f"seed_{opt.seed + opt.n_samples + opt.batch_size - 2}", f"{index:06d}_0_result.json")):
                continue
            seed_everything(opt.seed)

            outpath = os.path.join(opt.outdir, f"{index:0>5}")
            os.makedirs(outpath, exist_ok=True)

            prompt = metadata['prompt']
            n_rows = batch_size = opt.batch_size
            print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp)

            sample_count = 0

            with torch.no_grad():
                all_samples = list()
                for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                    current_sample_seed = opt.seed + n
                    
                    if opt.use_original_prompt:
                        sample = runner.run(async_generate_with_original_prompt(prompt, seed=current_sample_seed, 
                                                                                base_url=opt.base_url))
                    else:
                        sample = runner.run(async_generate_with_prompt(prompt, seed=current_sample_seed,
                                                    save_idx=index, saving_files=True,
                                                    saving_path=os.path.join(opt.outdir, "agent_results", f"seed_{current_sample_seed}"),
                                                    base_url=opt.base_url))

                    samples = [sample]
                    for sample in samples:
                        sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                        sample_count += 1
                    if not opt.skip_grid:
                        all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))
                    grid.save(os.path.join(outpath, f'grid.png'))
                    del grid
            del all_samples

        print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)

