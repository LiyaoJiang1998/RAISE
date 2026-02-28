"""Utility & helper functions."""

import io
import re
import base64
import json
from typing import List, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.load import dumpd
from PIL import Image
from langchain_ollama import ChatOllama

from graph.state import State

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str, seed=42, base_url="127.0.0.1:11434") -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    if provider == "ollama":
        return ChatOllama(base_url=base_url,
                          model=model, seed=seed,
                          temperature=0.15,
                          disable_streaming="tool_calling",
                          num_predict=-2,
                          num_ctx=16384)
    else:
        return init_chat_model(model, 
                               model_provider=provider, 
                               seed=seed)


def resize_to_max_resolution(input_image, max_resolution):
    """
    Resize image so that neither width nor height exceeds max_resolution,
    while preserving aspect ratio. Does not upscale smaller images.

    Args:
        input_image (PIL.Image): The input image object.
        max_resolution (int): Maximum allowed size for width or height.

    Returns:
        PIL.Image: The resized image.
    """
    width, height = input_image.size
    
    # scaling factor so that both width and height <= max_resolution
    scale = min(max_resolution / width, max_resolution / height, 1.0)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return input_image.resize((new_width, new_height), Image.LANCZOS)



def decode_data_url_to_base64(data_url: str) -> str:
    """
    Decode a base64 data URL (e.g., 'data:image/jpeg;base64,...')
    into a base64 string
    """
    # Extract mime type and base64 content
    match = re.match(r"^data:(.*?);base64,(.*)$", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid data URL format")

    mime_type, b64_data = match.group(1).lower(), match.group(2).strip()

    # Fix padding if necessary
    missing_padding = len(b64_data) % 4
    if missing_padding:
        b64_data += "=" * (4 - missing_padding)
    
    # Convert based on MIME type
    if mime_type.startswith("image/"):
        return b64_data
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")


def data_url_to_image(data_url: str) -> Image.Image:
    b64_data = decode_data_url_to_base64(data_url)
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def image_to_data_url(generated_image: Image.Image) -> str:
    import io
    import base64
    buf = io.BytesIO()
    generated_image.save(buf, format="PNG", optimize=True)
    mime = "image/png"
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def save_image_grid(
    images_rows: List[List[Image.Image]],
    save_path: str = "grid.jpg",
    padding: int = 10,                # horizontal space between images
    row_padding: int = 10,   # vertical space between rows
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    quality: int = 100,
    align: str = "left",              # "left", "center", or "right" row alignment
) -> Image.Image:
    """
    Combine images into a multi-row grid. Each inner list is a row.

    Args:
        images_rows: List of rows, each a list of PIL Images.
        save_path:   Output JPEG path.
        padding:     Horizontal pixels between images in a row (default 10).
        row_padding: Vertical pixels between rows. Defaults to same as padding.
        bg_color:    Background color (RGB).
        quality:     JPEG quality (1–100).
        align:       Row alignment on the canvas: "left", "center", or "right".

    Returns:
        The combined PIL.Image.
    """

    # Remove empty rows
    images_rows = [row for row in images_rows if row]
    if not images_rows:
        raise ValueError("No images provided (all rows are empty).")

    rows_resized, row_heights, row_widths = [], [], []

    # Process each row
    for row in images_rows:
        row_rgb = [img.convert("RGB") for img in row]
        row_max_h = max(img.height for img in row_rgb)
        resized_row = [
            im.resize(
                (int(im.width * row_max_h / im.height), row_max_h),
                Image.Resampling.LANCZOS
            )
            for im in row_rgb
        ]
        rows_resized.append(resized_row)
        row_widths.append(sum(im.width for im in resized_row) + padding * (len(resized_row) - 1))
        row_heights.append(row_max_h)

    canvas_w = max(row_widths)
    canvas_h = sum(row_heights) + row_padding * (len(row_heights) - 1)

    # Create the canvas
    new_img = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)

    # Paste rows
    y = 0
    for resized_row, row_w, row_h in zip(rows_resized, row_widths, row_heights):
        if align == "center":
            x = (canvas_w - row_w) // 2
        elif align == "right":
            x = canvas_w - row_w
        else:
            x = 0
        for im in resized_row:
            new_img.paste(im, (x, y))
            x += im.width + padding
        y += row_h + row_padding

    new_img.save(save_path, format="JPEG", quality=quality, subsampling=0, optimize=True)
    return new_img


def save_result(result, current_path_prefix: str, ident=4):
    """Saves the result dict to a JSON file, save images separately if any."""

    grid_images = [[] for _ in range(1+len(result["executor_outputs"]))]

    # Save the output image if any
    output_images = result.pop("output_images", None)
    if output_images:
        output_image_data_url = output_images[-1]
        output_image = data_url_to_image(output_image_data_url)
        output_image.save(f"{current_path_prefix}_0_output.jpg")
        grid_images[0].append(output_image)
        
    # Save each of the round best image
    round_best_images = result.pop("round_best_images", None)
    if round_best_images:
        for round_i in range(result["max_rounds"]):
            if round_i+1 > len(round_best_images):
                round_best_image_data_url = round_best_images[-1]
            else:
                round_best_image_data_url = round_best_images[round_i]
            round_best_image = data_url_to_image(round_best_image_data_url)
            round_best_image.save(f"{current_path_prefix}_best_of_round_{round_i+1}.jpg")

    for round_i, current_executor_output_list in enumerate(result["executor_outputs"]):
        for sample_j, current_executor_output in enumerate(current_executor_output_list):
            current_input_image_data_url = current_executor_output.pop("input_image_data_url", None)
            current_output_image_data_url = current_executor_output.pop("output_image_data_url", None)
            if current_input_image_data_url:
                current_input_image = data_url_to_image(current_input_image_data_url)
                current_input_image.save(f"{current_path_prefix}_{round_i+1}_input_sample{sample_j+1}.jpg")
            if current_output_image_data_url:   
                current_output_image = data_url_to_image(current_output_image_data_url)
                current_output_image.save(f"{current_path_prefix}_{round_i+1}_output_sample{sample_j+1}.jpg")
                grid_images[round_i+1].append(current_output_image)

    # Save the original prompt image if any
    save_original_prompt_image = result.pop("save_original_prompt_image", False)
    if save_original_prompt_image:
        original_prompt_image_data_url = result.pop("original_prompt_image", None)
        if original_prompt_image_data_url:
            original_prompt_image = data_url_to_image(original_prompt_image_data_url)
            original_prompt_image.save(f"{current_path_prefix}_0_original_prompt.jpg")
    
    if grid_images:
        save_image_grid(grid_images, f"{current_path_prefix}_0_grid.jpg", padding=10, row_padding=10)
    
    # Save the main result JSON
    result.pop("messages", None)
            
    output_file_path = f"{current_path_prefix}_0_result.json"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=ident)
    
    print(f"Result saved to {output_file_path}")

