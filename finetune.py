# =========================================================================================
#  Fine-tuning script based on https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb
# =========================================================================================

import os

from unsloth import FastLanguageModel, FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    standardize_sharegpt,
    train_on_responses_only,
)

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
import torch


my_dataset = os.getenv("DATASET", "mlabonne/FineTome-100k")
batch_size = int(os.getenv("BATCH_SIZE", 8))
learning_rate = float(os.getenv("LEARNING_RATE", 2e-5))
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 1))
max_steps = int(os.getenv("MAX_STEPS", 60))
full_finetuning = os.getenv("FULL_FINETUNING", "false").lower() == "true"
qat_scheme = os.getenv("QAT_SCHEME")
quantization_scheme = os.getenv("QUANTIZATION_SCHEME")
output_dir = os.getenv("OUTPUT_DIR", "/tmp")
model = os.getenv("MODEL", "Llama3.2-3B")


# =========================
#  Set up finetuning mode |
# =========================

if quantization_scheme is None:
    quantization_scheme = qat_scheme if qat_scheme is not None else "int4"
if qat_scheme is not None:
    assert qat_scheme == quantization_scheme

# qat scheme
if full_finetuning:
    qat_scheme_from_pretrained = qat_scheme
    qat_scheme_get_peft_model = None
else:
    qat_scheme_from_pretrained = None
    qat_scheme_get_peft_model = qat_scheme

# model
if model == "Llama3.2-1B":
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    chat_template = "llama-3.1"
elif model == "Llama3.2-3B":
    model_name = "unsloth/Llama-3.2-3B-Instruct"
    chat_template = "llama-3.1"
elif model == "Qwen3-8B":
    model_name = "unsloth/Qwen3-8B"
    chat_template = "qwen3"
elif model == "Qwen3-4B-Instruct":
    model_name = "unsloth/Qwen3-4B-Instruct-2507"
    chat_template = "qwen3-instruct"
elif model == "Gemma3-12B":
    model_name = "unsloth/gemma-3-12b-it"
    chat_template = "gemma3"
elif model == "Gemma3-4B":
    model_name = "unsloth/gemma-3-4b-it"
    chat_template = "gemma3"
else:
    raise ValueError(f"Unknown model {model}")

print("full_finetuning = ", full_finetuning)
print("qat_scheme = ", qat_scheme)
print("qat_scheme_from_pretrained = ", qat_scheme_from_pretrained)
print("qat_scheme_get_peft_model = ", qat_scheme_get_peft_model)
print("model_name = ", model_name)
print("chat_template = ", chat_template)
print("output_dir = ", output_dir)
print("dataset = ", my_dataset)


# ==============
#  Model setup |
# ==============

max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = False

if chat_template == "gemma3":
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        full_finetuning = full_finetuning,
        qat_scheme = qat_scheme_from_pretrained,
    )
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        full_finetuning = full_finetuning,
        qat_scheme = qat_scheme_from_pretrained,
    )

if not full_finetuning:
    if chat_template == "gemma3":
        # Taken from https://github.com/unslothai/notebooks/blob/c71d946696b0b69c191c4574d7ca1f99f077d2dc/nb/Gemma3_(4B).ipynb
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers     = False, # Turn off for just text!
            finetune_language_layers   = True,  # Should leave on!
            finetune_attention_modules = True,  # Attention good for GRPO
            finetune_mlp_modules       = True,  # SHould leave on always!
            r = 8,           # Larger = higher accuracy, but might overfit
            lora_alpha = 8,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
            qat_scheme = qat_scheme_get_peft_model,
        )
    else:
        # Taken from https://github.com/unslothai/notebooks/blob/c71d946696b0b69c191c4574d7ca1f99f077d2dc/nb/Kaggle-Llama3.2_(1B_and_3B)-Conversational.ipynb
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
            qat_scheme = qat_scheme_get_peft_model,
        )


# ============
#  Data prep |
# ============

# Format SlimOrca by dropping the "weights" attribute, which are not strings
# unsloth_zoo will error otherwise
if my_dataset == "Open-Orca/SlimOrca":
    def drop_weights_func(example):
        example["new_conversations"] = []
        for _dict in example["conversations"]:
            del _dict["weight"]
            example["new_conversations"].append(_dict)
        return example
    dataset = load_dataset(my_dataset, split = "train")
    dataset = dataset.map(drop_weights_func)
    dataset = dataset.remove_columns("conversations")
    dataset = dataset.rename_column("new_conversations", "conversations")
# Format Open-Platypus from "instruction" and "output" into a conversation
elif my_dataset == "garage-bAInd/Open-Platypus":
    def format_into_conversation(example):
        return {"conversations": [
            {"from": "human", "value": example["instruction"]},
            {"from": "gpt", "value": example["output"]},
        ]}
    dataset = load_dataset(my_dataset, split = "train")
    dataset = dataset.map(format_into_conversation)
    dataset = dataset.remove_columns(["input", "output", "instruction", "data_source"])
# Format mmlu from "question", "choices", and "answer" into a conversation
elif my_dataset == "cais/mmlu":
    def format_into_conversation(example):
        choices = ["A", "B", "C", "D"]
        correct_choice = choices[example["answer"]]
        question = "Choose the correct answer for the following question: "
        question += f"{example['question']}\n\n"
        question += "Choices:\n"
        question += f"A. {example['choices'][0]}\n"
        question += f"B. {example['choices'][1]}\n"
        question += f"C. {example['choices'][2]}\n"
        question += f"D. {example['choices'][3]}"
        answer = f"The correct answer is {correct_choice}."
        return {"conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]}
    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    dataset = dataset.map(format_into_conversation)
    dataset = dataset.remove_columns(["question", "subject", "choices", "answer"])
# Just leave FineTome-100k as is, it's already in a conversation format
elif my_dataset == "mlabonne/FineTome-100k":
    dataset = load_dataset(my_dataset, split = "train")
else:
    raise ValueError(f"Unknown dataset {my_dataset}")

# Standardize dataset format
data_collator = None
tokenizer = get_chat_template(tokenizer, chat_template = chat_template)
if "llama" in chat_template:
    dataset = standardize_sharegpt(dataset)
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
else:
    dataset = standardize_data_formats(dataset)

# Format conversation into text prompt using the correct chat template based on the model
if chat_template == "gemma3":
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }
else:
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
dataset = dataset.map(formatting_prompts_func, batched = True,)

# Print the first example for sanity check
print(f"First dataset example:\n\n{dataset[0]}")


# ========
#  Train |
# ========

if max_steps != 0:
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = data_collator,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = 5,
            num_train_epochs = 1,
            max_steps = max_steps,
            learning_rate = learning_rate,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    if "llama" in chat_template:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif "qwen3" in chat_template:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )
    elif chat_template == "gemma3":
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<start_of_turn>user\n",
            response_part = "<start_of_turn>model\n",
        )
    else:
        raise ValueError(f"Unknown chat template {chat_template}")

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# =============
#  Model save |
# =============

# Save high precision models
if full_finetuning:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
else:
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
