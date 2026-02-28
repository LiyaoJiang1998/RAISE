import PIL
from transformers import AutoConfig, AutoModel, set_seed



MODEL_NVILA = None
YES_ID = None
NO_ID = None



def get_nvila_model():
    global MODEL_NVILA, YES_ID, NO_ID
    
    if MODEL_NVILA is None:
        model_path = "Efficient-Large-Model/NVILA-Lite-2B-Verifier"
        MODEL_NVILA = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        
        YES_ID = MODEL_NVILA.tokenizer.encode("yes", add_special_tokens=False)[0]
        NO_ID = MODEL_NVILA.tokenizer.encode("no", add_special_tokens=False)[0]

        

def evaluate_nvila_score(image, prompt, seed):
    prompt = f"""You are an AI assistant specializing in image analysis and ranking. Your task is to analyze and compare image based on how well they match the given prompt. 
The given prompt is:{prompt}. Please consider the prompt and the image to make a decision and response directly with 'yes' or 'no'.
"""

    if MODEL_NVILA is None:
        get_nvila_model()
        
    set_seed(seed)
    yes_or_no_str, scores = MODEL_NVILA.generate_content([image, prompt])
    
    if yes_or_no_str == "yes":
        score = float(scores[0][0, YES_ID])
    else:
        score = -float(scores[0][0, NO_ID])
    
    return score
    
    
