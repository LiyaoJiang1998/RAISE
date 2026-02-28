"""Define the configurable parameters for the agent."""

from __future__ import annotations
import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="ollama/mistral-small3.2:24b-instruct-2506-fp16",
        # default="ollama/qwen3-vl:32b-instruct-bf16",
        # default="ollama/gemma3:27b-it-fp16",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )
    
    base_url: str = field(
        default="127.0.0.1:11434",
        metadata={
            "description": "The base URL for the Ollama model API. "
        },
    )

    seed: int = field(
        default=42,
        metadata={
            "description": "The random seed to use for Chat Model's reproducibility."
        },
    )
    
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    
    system_prompt_analyzer_generation: str = field(
        default=prompts.SYSTEM_PROMPT_ANALYZER_GENERATION,
        metadata={
            "description": "The system prompt to use for the generation analyzer agent's interactions. "
        },
    )
        
    system_prompt_rewriter_generation: str = field(
        default=prompts.SYSTEM_PROMPT_REWRITER_GENERATION,
        metadata={
            "description": "The system prompt to use for the generation rewriter agent's interactions. "
        },
    )
    
    system_prompt_rewriter_editing: str = field(
        default=prompts.SYSTEM_PROMPT_REWRITER_EDITING,
        metadata={
            "description": "The system prompt to use for the editing rewriter agent's interactions. "
        },
    )
    
    system_prompt_verifier_generation: str = field(
        default=prompts.SYSTEM_PROMPT_VERIFIER_GENERATION,
        metadata={
            "description": "The system prompt to use for the generation verifier agent's interactions. "
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
