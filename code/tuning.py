# Tunes a bot given a generated dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import gc

serversDirectory = "discord_servers"

trainer = None
model = None
tokenizer = None

def getServerPaths(guild_id):
    guildDir = f"{serversDirectory}/{guild_id}"
    return {
        "guild": guildDir,
        "history": f"{guildDir}/history.json",
        "users": f"{guildDir}/users.json",
        "tuning": f"{guildDir}/training_data",
        "adapters": f"{guildDir}/adapters"
    }

def formatting_func(examples):
    convos = examples["conversations"]
    if isinstance(convos[0], dict):
        convos = [convos]
    return [
        tokenizer.apply_chat_template(c, tokenize=False, enable_thinking=False)
        for c in convos
    ]

def tuneModel(guildID, username):
    global trainer, model, tokenizer
    try:
        path = f"{getServerPaths(guildID)['tuning']}/{username}_training_data.json"
        dataset = load_dataset("json", data_files={"train": [path]})
        dataset = dataset["train"]
        dataset = standardize_sharegpt(dataset)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen3-8B",
            max_seq_length = 4096,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            #USED TO BE 0
            lora_dropout = 0.05,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 42,
        )

        tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
    

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args = SFTConfig(
                max_length=4096,
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 8,
                warmup_ratio = 0.05,
                num_train_epochs = 2,
                learning_rate = 5e-5,
                logging_steps = 10,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 42,
                output_dir = "outputs",
            ),
        )

        trainer.train()
        model.save_pretrained(f"{getServerPaths(guildID)['adapters']}/{username}")
        tokenizer.save_pretrained(f"{getServerPaths(guildID)['adapters']}/{username}")
    except:
        print("ERROR IN TUNING!")
    finally:
        del trainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
