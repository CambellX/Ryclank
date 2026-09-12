from unsloth import FastLanguageModel
import torch
from pathlib import Path
import re
import gc

adapter_path = None
current_username = None
tokenizer = None
model = None

# Setup so the chatbot can access the adapters directory later.
serversDirectory = "discord_servers"

def getServerPaths(guild_id):
    guildDir = f"{serversDirectory}/{guild_id}"
    return {
        "guild": guildDir,
        "history": f"{guildDir}/history.json",
        "users": f"{guildDir}/users.json",
        "tuning": f"{guildDir}/training_data",
        "adapters": f"{guildDir}/adapters"
    }

# Equips a LoRA adapter givven the username and Discord server ID
def equipAdapter(guild_id, username):
    global adapter_path, current_username, tokenizer, model
    # Get the adapter path
    adapterPath = f"{getServerPaths(guild_id)['adapters']}/{username}"
    if not Path(adapterPath).is_dir():
        return False
    adapter_path = adapterPath
    current_username = username
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return True

# Frees all the memory used by the model.
def unloadLLM():
    global adapter_path, current_username, tokenizer, model

    model = tokenizer = adapter_path = current_username = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def makeSystemPrompt(username):
    return (
        f"""
        You are {username}.
        You are responding to a conversation between your friends in a shared Discord groupchat.
        You insult each other regularly, curse, say inappropriate things, be rude, act stupid, and
        talk about your day. Everything is in good fun, so remember to joke and
        laugh, and be funny. You will be given the most recent messages in the group chat.
        Write a response to the conversation as {username}."""
    )

def makeUserPrompt(conversation, username):
    return f"""
    Recent conversation:
    {conversation}
    
    Reply to the last message as {username}:
    """

# _SPEAKER_LINE = re.compile(r"^[A-Za-z0-9_.-]+:\s")
# def _cleanResponse(raw, username):
#     text = raw.strip()
#     prefix = f"{username}:"
#     if text.lower().startswith(prefix.lower()):
#         text = text[len(prefix):].lstrip()
#     lines = text.splitlines()
#     for i, line in enumerate(lines):
#         if i and _SPEAKER_LINE.match(line):
#             return "\n".join(lines[:i]).strip()
#     return text

def _cleanResponse(raw, username):
    text = raw.strip()
    prefix = f"{username}:"
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].lstrip()
    return text

def sendMessage(conversation, latest_author=None, latest_message=None):
    if adapter_path is None:
        return "No adapter equipped"

    if latest_author and latest_message:
        latest_line = f"{latest_author}: {latest_message}"
        if not conversation.strip().endswith(latest_message.strip()):
            conversation = f"{conversation}\n{latest_line}" if conversation else latest_line

    messages = [
        {"role": "system", "content": makeSystemPrompt(current_username)},
        {"role": "user", "content": makeUserPrompt(conversation, current_username)},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    text = f"{text}{current_username}:"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        top_k=40,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    raw = tokenizer.decode(output_ids, skip_special_tokens=True)
    return _cleanResponse(raw, current_username)

