import os
import argparse
from typing import List, Union

import numpy as np
import cv2
from PIL import Image
import torch
import supervision as sv

from transformers import AutoProcessor, AutoModelForCausalLM
from grounded_sam_2.sam2.build_sam import build_sam2
from grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from grounded_sam_2.utils.supervision_utils import CUSTOM_COLOR_MAP



"""
Reference:
- Caption and Grounding Code Adapted from: https://github.com/IDEA-Research/Grounded-SAM-2
- Depth Estimation Code using MiDaS: https://github.com/isl-org/MiDaS, https://pytorch.org/hub/intelisl_midas_v2
"""



TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

CAPTION_TO_TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>"
}

SAM2_MODEL = None
SAM2_PREDICTOR = None
FLORENCE2_MODEL = None
FLORENCE2_PROCESSOR = None
MIDAS_DEPTH_ESTIMATOR = None



def average_depth_per_mask_tensor(
    depth_pil: Image.Image,
    masks: np.ndarray,
    *,
    allow_nan: bool = False,
    empty_value: int = -1,
) -> List[Union[int, float]]:
    """
    Compute the average depth for each mask in a CxHxW mask tensor.

    Args:
        depth_pil: PIL 'L' depth image (HxW), 0–255.
        masks: np.ndarray of shape (C, H, W), binary or boolean.
        allow_nan: if True -> returns float list, with np.nan for empty masks.
                   if False -> returns int list, with `empty_value` for empty masks.
        empty_value: integer sentinel if allow_nan=False and a mask is empty.

    Returns:
        List[int] or List[float]: average depth per mask.
    """
    # Ensure depth is grayscale uint8
    if depth_pil.mode != "L":
        depth_pil = depth_pil.convert("L")
    depth = np.array(depth_pil, dtype=np.uint8)  # (H, W)

    C, H, W = masks.shape
    if depth.shape != (H, W):
        raise ValueError(f"Depth image shape {depth.shape} does not match mask shape {(H, W)}")

    # Boolean mask
    masks_bool = masks.astype(bool)

    # Flatten for vectorized computation
    depth_flat = depth.reshape(1, -1)           # (1, H*W)
    masks_flat = masks_bool.reshape(C, -1)      # (C, H*W)

    pixel_counts = masks_flat.sum(axis=1)       # (C,)

    if allow_nan:
        # Use float32 and np.nan for empty masks
        avg_depths = np.full(C, np.nan, dtype=np.float32)
        valid = pixel_counts > 0
        if valid.any():
            sums = (masks_flat[valid] * depth_flat).sum(axis=1)
            avg_depths[valid] = sums / pixel_counts[valid]
        return avg_depths.tolist()
    else:
        # Use int32 and a sentinel for empty masks
        avg_depths = np.full(C, empty_value, dtype=np.int32)
        valid = pixel_counts > 0
        if valid.any():
            sums = (masks_flat[valid] * depth_flat).sum(axis=1)
            avg_depths[valid] = np.round(sums / pixel_counts[valid]).astype(np.int32)
        return avg_depths.tolist()



class MidasDepthEstimator:
    """
    Robust MiDaS wrapper: PIL.Image -> 1-channel PIL.Image ('L')
    - Handles both transform APIs: returns dict or tensor
    - Handles input as PIL or NumPy
    """
    def __init__(self, model_type: str = "DPT_Large", device: str | None = None):
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.inference_mode()
    def __call__(self, img_pil: Image.Image, invert_for_visual: bool = True) -> Image.Image:
        """
        Returns: 1-channel PIL 'L' image with values in [0,255].
        invert_for_visual=True -> nearer appears brighter.
        """
        # Keep original size
        orig_w, orig_h = img_pil.size
        img_rgb = img_pil.convert("RGB")

        # ---- Robust transform handling (dict or tensor; PIL or NumPy) ----
        inp_any = None
        try:
            # Most hubs accept PIL directly
            inp_any = self.transform(img_rgb)
        except Exception:
            # Fallback: some transforms expect NumPy
            inp_any = self.transform(np.array(img_rgb))

        if isinstance(inp_any, dict):
            inp = inp_any["image"]
        elif torch.is_tensor(inp_any):
            inp = inp_any
        elif isinstance(inp_any, (list, tuple)) and len(inp_any) > 0:
            # Very rare: choose first tensor-like item
            inp = inp_any[0] if torch.is_tensor(inp_any[0]) else inp_any
            if isinstance(inp, dict):
                inp = inp["image"]
        else:
            raise TypeError(f"Unexpected transform output type: {type(inp_any)}")

        # Ensure shape [1,3,H,W], dtype float32, on device
        if inp.ndim == 3:         # [3,H,W]
            inp = inp.unsqueeze(0)
        elif inp.ndim != 4:       # anything else is invalid for the model
            raise ValueError(f"Unexpected input tensor shape from transform: {tuple(inp.shape)}")
        inp = inp.to(self.device, dtype=torch.float32, non_blocking=True)

        # ---- Forward pass ----
        pred = self.model(inp)  # shapes vary: [1,H',W'] or [1,1,H',W']

        # Normalize shapes to [1,1,H',W'] for interpolate
        if pred.ndim == 2:             # [H',W']
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 3:           # [1,H',W']
            pred = pred.unsqueeze(1)
        elif pred.ndim == 4:
            if pred.shape[1] != 1:     # sometimes [1,?,H',W'] — take channel 0
                pred = pred[:, :1, ...]
        else:
            raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")

        # Resize to original image size
        pred = torch.nn.functional.interpolate(
            pred, size=(orig_h, orig_w), mode="bicubic", align_corners=False
        )[0, 0]  # -> [H,W]

        # ---- Min-max normalize to 0..255 for viewable 8-bit output ----
        d = pred.float().cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        if invert_for_visual:
            d = 1.0 - d  # optional: nearer = brighter
        d_u8 = (d * 255.0).clip(0, 255).astype(np.uint8)

        return Image.fromarray(d_u8, mode="L")



def get_aux_models():
    """
    Init Florence-2 and SAM 2 Model
    """
    FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
    SAM2_CHECKPOINT = "./src/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    global SAM2_MODEL, SAM2_PREDICTOR, FLORENCE2_MODEL, FLORENCE2_PROCESSOR, MIDAS_DEPTH_ESTIMATOR
    if SAM2_MODEL is None:
        # environment settings
        # use bfloat16
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # build florence-2
        FLORENCE2_MODEL = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
        FLORENCE2_PROCESSOR = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

        # build sam 2
        SAM2_MODEL = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        SAM2_PREDICTOR = SAM2ImagePredictor(SAM2_MODEL)

        MIDAS_DEPTH_ESTIMATOR = MidasDepthEstimator(device=device)

        return SAM2_MODEL, SAM2_PREDICTOR, FLORENCE2_MODEL, FLORENCE2_PROCESSOR, MIDAS_DEPTH_ESTIMATOR
    else:
        return SAM2_MODEL, SAM2_PREDICTOR, FLORENCE2_MODEL, FLORENCE2_PROCESSOR, MIDAS_DEPTH_ESTIMATOR

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer



def run_caption_ground_depth(image, 
                       florence2_model, florence2_processor, sam2_model, sam2_predictor, midas_depth_estimator,
                       caption_type="more_detailed_caption"):
    
    assert caption_type in ["caption", "detailed_caption", "more_detailed_caption"]
    caption_task_prompt = CAPTION_TO_TASK_PROMPT[caption_type]
    assert caption_task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
    
    # image caption
    caption_results = run_florence2(caption_task_prompt, None, florence2_model, florence2_processor, image)
    caption_text = caption_results[caption_task_prompt]
    
    # phrase grounding
    grounding_results = run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', caption_text, florence2_model, florence2_processor, image)
    grounding_results = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
    
    # parse florence-2 detection results
    input_boxes = np.array(grounding_results["bboxes"])
    class_names = grounding_results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    # # normalize boxes
    # normalized_boxes = input_boxes.copy()
    # normalized_boxes[:, [0, 2]] /= image_width   # x_min, x_max
    # normalized_boxes[:, [1, 3]] /= image_height  # y_min, y_max
    
    # integer boxes
    integer_boxes = [[int(value) for value in input_box] for input_box in input_boxes]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    annotated_image_cv2 = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_image_cv2 = label_annotator.annotate(scene=annotated_image_cv2, detections=detections, labels=labels)
    annotated_image_cv2 = mask_annotator.annotate(scene=annotated_image_cv2, detections=detections)
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_image_cv2, cv2.COLOR_BGR2RGB))
    
    # Depth Estimation
    depth_image = midas_depth_estimator(image)
    avg_depths = average_depth_per_mask_tensor(depth_image, masks)

    return caption_text, integer_boxes, labels, avg_depths, annotated_image, depth_image



def run_referring_expression_segmentation(
    image, text_input,
    florence2_model,
    florence2_processor,
    sam2_model,
    sam2_predictor,
):
    '''
    Mark text_input region as transparent red box on the image using referring expression segmentation.
    '''
    task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    assert text_input is not None, "Text input should not be None when calling referring segmentation pipeline."
    results = results[task_prompt]
    # parse florence-2 detection results
    polygon_points = np.array(results["polygons"][0], dtype=np.int32).reshape(-1, 2)
    class_names = [text_input]
    class_ids = np.array(list(range(len(class_names))))
    
    # parse polygon format to mask
    img_width, img_height = image.size[0], image.size[1]
    florence2_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    if len(polygon_points) < 3:
        print("Invalid polygon:", polygon_points)
        return image # return original image if polygon is invalid
    
    cv2.fillPoly(florence2_mask, [polygon_points], 1)
    if florence2_mask.ndim == 2:
        florence2_mask = florence2_mask[None]

    # compute bounding box based on polygon points
    x_min = np.min(polygon_points[:, 0])
    y_min = np.min(polygon_points[:, 1])
    x_max = np.max(polygon_points[:, 0])
    y_max = np.max(polygon_points[:, 1])

    input_boxes = np.array([[x_min, y_min, x_max, y_max]])

    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    sam2_masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if sam2_masks.ndim == 4:
        sam2_masks = sam2_masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # # florence2 mask
    # detections = sv.Detections(
    #     xyxy=input_boxes,
    #     mask=florence2_mask.astype(bool),
    #     class_id=class_ids
    # )
    # sam2 mask
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=sam2_masks.astype(bool),
        class_id=class_ids
    )

    # Paint the area inside the mask with a specific color (e.g., red)
    color_annotator = sv.ColorAnnotator(color=sv.Color(r=255, g=0, b=0), opacity=0.5)
    annotated_image_cv2 = color_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_image_cv2, cv2.COLOR_BGR2RGB))
    
    return annotated_image



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Auxiliary Models for Caption and Grounding", add_help=True)
    parser.add_argument("--image_path", type=str, default="./notebooks/images/cars.jpg", required=True, help="path to image file")
    parser.add_argument("--caption_type", type=str, default="more_detailed_caption", choices=["caption", "detailed_caption", "more_detailed_caption"], required=False, help="granularity of caption")
    parser.add_argument("--output_dir", type=str, default="./outputs", required=False, help="output directory to save results")
    
    args = parser.parse_args()
        
    image = Image.open(args.image_path).convert("RGB")

    SAM2_MODEL, SAM2_PREDICTOR, FLORENCE2_MODEL, FLORENCE2_PROCESSOR, MIDAS_DEPTH_ESTIMATOR = get_aux_models()
    caption_text, integer_boxes, labels, avg_depths, annotated_image, depth_image = run_caption_ground_depth(
        image=image,
        florence2_model=FLORENCE2_MODEL, florence2_processor=FLORENCE2_PROCESSOR,
        sam2_model=SAM2_MODEL, sam2_predictor=SAM2_PREDICTOR, 
        midas_depth_estimator=MIDAS_DEPTH_ESTIMATOR,
        caption_type=args.caption_type
    )

    # labels and boxes
    for label, box in zip(labels, integer_boxes):
        print(f"Label: {label}, Box: {box}, Avg Depth: {avg_depths[labels.index(label)]}")
    
    # image size
    image_width, image_height = image.size
    print("Image (width, height): ", (image_width, image_height))
    
    # image caption
    print(f'Aux Model Generated Caption: ', caption_text)
    
    annotated_image = run_referring_expression_segmentation(
        image=image,
        text_input="the sandwich",
        florence2_model=FLORENCE2_MODEL, florence2_processor=FLORENCE2_PROCESSOR,
        sam2_model=SAM2_MODEL, sam2_predictor=SAM2_PREDICTOR,
    )
    # annotated_image.save(os.path.join(args.output_dir, "referring_expression_segmentation_output.png"))
    