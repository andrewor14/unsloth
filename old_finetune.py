import os
from unsloth import FastLanguageModel
import torch


batch_size = int(os.getenv("BATCH_SIZE", 2))
learning_rate = float(os.getenv("LEARNING_RATE", 2e-5))
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 4))
max_steps = int(os.getenv("MAX_STEPS", 60))
full_finetuning = os.getenv("FULL_FINETUNING", "true").lower() == "true"
qat_scheme = os.getenv("QAT_SCHEME")
quantization_scheme = os.getenv("QUANTIZATION_SCHEME")
save_output_dir = os.getenv("SAVE_OUTPUT_DIR", "/tmp")
model = os.getenv("MODEL", "Llama3.1-8B")


# =========================
#  Set up finetuning mode |
# =========================

if quantization_scheme is None:
    quantization_scheme = qat_scheme if qat_scheme is not None else "fp8-int4"
if qat_scheme is not None:
    assert qat_scheme == quantization_scheme

# output path
save_output_path = f"{save_output_dir}/unsloth_model"
if full_finetuning:
    save_output_path += "_full"
else:
    save_output_path += "_lora"
if qat_scheme is not None:
    save_output_path += f"_qat_{qat_scheme}"
else:
    save_output_path += "_baseline"
save_output_path += "_output"

# qat scheme
if full_finetuning:
    qat_scheme_from_pretrained = qat_scheme
    qat_scheme_get_peft_model = None
else:
    qat_scheme_from_pretrained = None
    qat_scheme_get_peft_model = qat_scheme

# model
if model == "Llama3.1-8B":
    model_name = "unsloth/Meta-Llama-3.1-8B"
elif model == "Qwen3-8B":
    model_name = "unsloth/Qwen3-8B"
else:
    raise ValueError(f"Unknown model {model}")

print("full_finetuning = ", full_finetuning)
print("qat_scheme = ", qat_scheme)
print("qat_scheme_from_pretrained = ", qat_scheme_from_pretrained)
print("qat_scheme_get_peft_model = ", qat_scheme_get_peft_model)
print("model_name = ", model_name)
print("save_output_path = ", save_output_path)


# ==============
#  Model setup |
# ==============

max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    full_finetuning = full_finetuning,
    qat_scheme = qat_scheme_from_pretrained,
)

if not full_finetuning:
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

# Pass dummy inputs
#from transformers import AutoTokenizer
#model_name = "Qwen/Qwen3-8B"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#dummy_text = "This is a dummy input for the Qwen3 model."
#inputs = tokenizer(dummy_text, return_tensors="pt").to("cuda")
#print("Dummy inputs", inputs)
#
#print("\n\n\n\n======= Calling model")
#model(**inputs)
#print("\n\n\n\n======= Calling model.base_model")
#model.base_model(**inputs)

# ============
#  Data prep |
# ============

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# ========
#  Train |
# ========

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = max_steps,
        learning_rate = learning_rate,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

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

#from torchao.quantization import (
#    Float8DynamicActivationInt4WeightConfig,
#    Int8DynamicActivationInt4WeightConfig,
#)
#from unsloth.save import patch_saving_functions
#
## Save quantized models first
#save_quantized_output_path = save_output_path + "_quantized"
#patch_saving_functions(model)
#if quantization_scheme == "fp8-int4":
#    torchao_config = Float8DynamicActivationInt4WeightConfig()
#elif quantization_scheme == "int8-int4":
#    torchao_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
#else:
#    raise ValueError("unknown qat scheme")
#model.save_pretrained_torchao(save_quantized_output_path, tokenizer, torchao_config)

# Save high precision models
if full_finetuning:
    model.save_pretrained(save_output_path)
    tokenizer.save_pretrained(save_output_path)
else:
    model.save_pretrained_merged(save_output_path, tokenizer, save_method="merged_16bit")
