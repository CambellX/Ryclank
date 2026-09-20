from unsloth import FastLanguageModel
import torch
import time
import re
import gc

model = None
tokenizer = None

# Loads a convo model 
def loadConversationModel():
    global model, tokenizer
    if model is not None:
        return
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length = 4096,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

# on_message stores as a string, but setup stores as dicts.
# function converts everything into a standard format.
# The input will either be a conversations array of messages or a single message.
def toString(input):
    # if the input is a string, then just return it
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
    # if the input is a dict, convert it to string format
        return f"{input.get('author')}: {input.get('content')}"
    return "\n".join(toString(i) for i in input)

@torch.inference_mode()
def separateConversations(currentConversation, nextMessage):
    currentConversation = toString(currentConversation)
    nextMessage = toString(nextMessage)
    messages = [
    {
        "role": "system",
        "content": """You classify conversation boundaries in a discord group chat.

            You are given the CURRENT CONVERSATION and the NEXT MESSAGE. Decide how the next message relates to the current conversation and answer with one label:

            - CONTINUE: it replies to, reacts to, or continues the current topic.
            - NEW: it starts a different, unrelated topic.
            - UNCERTAIN: there isn't enough information to tell.

            Reaction replies like "yeah", "huh", or "holy" usually continue the topic.
            Answer only with one label.""",
    },
    {
        "role": "user",
        "content": f"""
        CURRENT CONVERSATION:
        {currentConversation}

        NEXT MESSAGE:
        {nextMessage}

        Classification:
        """,
    },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated_ids[0, model_inputs.input_ids.shape[1]:]
    content = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip().upper()

    match = re.search(r"\b(CONTINUE|NEW|UNCERTAIN)\b", content)
    if not match:
        print(f"CONVERSATION PARSING ISSUE: {content}")
        return "UNCERTAIN"
    return match.group(1)

def unloadConversationModel():
    global model, tokenizer
    model = None
    tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()