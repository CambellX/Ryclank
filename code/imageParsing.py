# Allows the bot to read and react to images. Will be used later when it will be implemented
# but rn just used for training data.
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import time
import torch
import gc

model = None
processor = None

def loadImagingModel():
    # default: Load the model on the available device(s)
    global model, processor
    if model is not None:
        return
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen3-VL-8B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

@torch.inference_mode()
def processImage(imageURL):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": imageURL,
                },
                {
                "type": "text", "text": """
                You are looking at an image from a groupchat of friends. Describe the image using the following format:
                
                Genre: (meme, screenshot, news, etc)
                Text_in_image: (Copy the main readable text, up to ~40 words or "None". Do not repeat lines)
                Description: (1-3 sentences on what is shown and what stands out)
                Tone: (Funny, angry, sad, ragebait, neutral, unknown, other)
                """},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=False,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.eos_token_id,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def unloadImagingModel():
    global model, processor
    model = None
    processor = None
    gc.collect()
    torch.cuda.empty_cache()