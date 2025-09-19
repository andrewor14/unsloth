# =========================================================================================
#  Inference script based on https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb
# =========================================================================================

import os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig, TextStreamer
from torchao.quantization import (
    Float8DynamicActivationInt4WeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
)


model = os.getenv("MODEL", "Llama3.2-3B")
model_dir = os.getenv("MODEL_DIR")
if model_dir is None:
    raise ValueError("MODEL_DIR must be set")
quantized_model_dir = model_dir + "_quantized2"

# Quantization config
quantization_scheme = os.getenv("QUANTIZATION_SCHEME")
if quantization_scheme is not None:
    if quantization_scheme == "fp8-int4":
        ao_config = Float8DynamicActivationInt4WeightConfig()
    elif quantization_scheme == "fp8-fp8":
        ao_config = Float8DynamicActivationFloat8WeightConfig()
    elif quantization_scheme == "int8-int4":
        ao_config = Int8DynamicActivationInt4WeightConfig()
    elif quantization_scheme == "int4":
        ao_config = Int4WeightOnlyConfig()
    else:
        raise ValueError(f"Unknown quantization scheme {quantization_scheme}")
    quantization_config = TorchAoConfig(ao_config)
else:
    quantization_config = None

# Chat template
if model == "Llama3.2-1B":
    chat_template = "llama-3.1"
elif model == "Llama3.2-3B":
    chat_template = "llama-3.1"
elif model == "Qwen3-8B":
    chat_template = "qwen3"
elif model == "Qwen3-4B-Instruct":
    chat_template = "qwen3-instruct"
elif model == "Gemma3-12B":
    chat_template = "gemma3"
elif model == "Gemma3-4B":
    chat_template = "gemma3"
else:
    raise ValueError(f"Unknown model {model}")


# ==========================
#  Load and quantize model |
# ==========================

print("Running fibonacci on model saved in ", model_dir, "using quantization scheme? ", quantization_scheme)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Model: ", model)
if "llama" in chat_template or "qwen" in chat_template:
    print("First linear weight: ", model.model.layers[0].self_attn.q_proj.weight)
elif "gemma" in chat_template:
    print("First linear weight: ", model.model.language_model.layers[0].self_attn.q_proj.weight)

print("Saving model and tokenizer to ", quantized_model_dir)

model.save_pretrained(quantized_model_dir, safe_serialization=False)
tokenizer.save_pretrained(quantized_model_dir)


# ============
#  Inference |
# ============

tokenizer = get_chat_template(tokenizer, chat_template = chat_template)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]

if "llama" in chat_template:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                       use_cache = True, temperature = 1.5, min_p = 0.1)
elif "qwen3" in chat_template:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 1000, # Increase for longer outputs!
        temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
elif chat_template == "gemma3":
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )
    _ = model.generate(
        **tokenizer([text], return_tensors = "pt").to("cuda"),
        max_new_tokens = 64, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
else:
    raise ValueError(f"Unknown chat template {chat_template}")
