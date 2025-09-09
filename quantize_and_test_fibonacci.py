import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig, TextStreamer
from torchao.quantization import (
    Float8DynamicActivationInt4WeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
)

model_dir = os.getenv("MODEL_DIR")
if model_dir is None:
    raise ValueError("MODEL_DIR must be set")
quantized_model_dir = model_dir + "_quantized2"

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

print("Running fibonacci on model saved in ", model_dir, "using quantization scheme? ", quantization_scheme)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Model: ", model)
print("First linear weight: ", model.model.layers[0].self_attn.q_proj.weight)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

print("Saving model and tokenizer to ", quantized_model_dir)

model.save_pretrained(quantized_model_dir, safe_serialization=False)
tokenizer.save_pretrained(quantized_model_dir)
