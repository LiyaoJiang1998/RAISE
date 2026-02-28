import random
import time
import asyncio
from contextlib import suppress

import numpy as np
import langchain_core
from langgraph.runtime import get_runtime
from graph.utils import load_chat_model

MAX_SEED = np.iinfo(np.int32).max

CHAT_MODEL = None
CHAT_MODEL_SEED = None


def get_chat_model(model, seed, base_url):
    global CHAT_MODEL, CHAT_MODEL_SEED

    start_time = time.time()
    if seed == CHAT_MODEL_SEED and CHAT_MODEL_SEED is not None and CHAT_MODEL is not None:
        print_load = False
    else:
        CHAT_MODEL = load_chat_model(fully_specified_name=model, seed=seed, base_url=base_url)
        CHAT_MODEL_SEED = seed
        CHAT_MODEL.invoke("")  # warm up the model for reproducibility
        print_load = True
    end_time = time.time()
    
    duration = "%.4f" % (end_time - start_time)
    if print_load:
        print("Model loaded:", model, "| with seed:", seed, "| Time taken:", duration, "seconds")
    
    return CHAT_MODEL


async def ainvoke_with_timeout_retry(state, runtime, inp, output_class=None, timeout=60, retries=20, backoff=0.6, shield=False):
    for attempt in range(retries + 1):
        # Get the model. For attempt > 0, recreate the model with a different seed to avoid infinite output
        seed = runtime.context.seed + attempt
        
        model = get_chat_model(model=runtime.context.model, 
                               seed=seed,
                               base_url=runtime.context.base_url) 
        if output_class:
            model = model.with_structured_output(output_class)
        
        # Create the task
        if attempt > 0:            
            print(f"Retrying... attempt {attempt} of {retries}")
            inp[0]["content"] += f"\n\nRetry attempt: {attempt} out of {retries}. Your previous response was repeating and infinite. Please be concise, do not output infinite repeated content or a long list. Follow the output format"    
            task = asyncio.create_task(model.ainvoke(inp))
        else:
            task = asyncio.create_task(model.ainvoke(inp))
        
        # Run the task
        try:
            coro = asyncio.wait_for(task, timeout=timeout)
            return await (asyncio.shield(coro) if shield else coro)
        
        except asyncio.TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task  # ensure proper cleanup
            if attempt == retries:
                raise
            await asyncio.sleep(backoff * (2 ** attempt))  # simple exp backoff
        
        except langchain_core.exceptions.OutputParserException:
            if attempt == retries:
                raise
            await asyncio.sleep(backoff * (2 ** attempt))  # simple exp backoff
        
        except Exception as e:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise

