"""Define the state structures for the agent."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import List, Sequence, Literal
from operator import add

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from PIL import Image


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """
    
    # shared settings
    agent_setup: Literal['RAISE', 'original'] = 'RAISE'
    saving_files: bool = field(default=False)
    saving_path: str = field(default="outputs/examples")
    image_seed: int = field(default=42)
    vlm_max_resolution:  int = field(default=1024)
    
    # Image Generation Settings
    gen_num_inference_steps: int = field(default=28) # FLUX.1-dev
    gen_guidance_scale: float = field(default=3.5) # FLUX.1-dev
    
    gen_num_inference_steps_speed: int = field(default=4) # FLUX.1-schnell
    gen_guidance_scale_speed: float = field(default=0) # FLUX.1-schnell
    # gen_num_inference_steps_speed: int = field(default=20) # SANA1.5_4.8B
    # gen_guidance_scale_speed: float = field(default=4.5) # SANA1.5_4.8B
    
    gen_height: int = field(default=1024)
    gen_width: int = field(default=1024)
    
    # Image Editing Settings
    edit_num_inference_steps: int = field(default=28)
    edit_guidance_scale: float = field(default=2.5)
    edit_max_resolution: int = field(default=1024)


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """
    # Additional attributes can be added here as needed.
    output_images: Annotated[List[str], add] = field(default_factory=list)
    round_best_images: Annotated[List[str], add] = field(default_factory=list) # list of best image upto each round
    
    save_original_prompt_image: bool = field(default=False)
    original_prompt_image: str = field(default=None)
    
    analyzer_outputs: Annotated[List[dict], add] = field(default_factory=list)
    rewriter_outputs: Annotated[List[dict], add] = field(default_factory=list)
    executor_outputs: Annotated[List[List[dict]], add] = field(default_factory=list)
    verifier_outputs: Annotated[List[List[dict]], add] = field(default_factory=list)
    
    original_prompt: str = field(default=None)
    
    current_round: int = field(default=1) # current round index, starting from 1
    max_rounds: int = field(default=4) # maximum number of rounds allowed, including the first round
    min_rounds: int = field(default=2) # minimum number of rounds allowed, including the first round
    best_round: int = field(default=1) # Initialized to 1, will be updated to the best round index (1-based)
    best_round_sample: int = field(default=1) # Initialized to 1, will be updated to the best sample index within the best round (1-based)

    use_caption_and_grounding: bool = field(default=True)
    