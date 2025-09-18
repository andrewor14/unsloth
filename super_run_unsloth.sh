BASE_LOG_DIR="${BASE_LOG_DIR:-/home/andrewor/local/logs/unsloth}"
ENABLE_QAT="${ENABLE_QAT:-true}"
export FULL_FINETUNING="${FULL_FINETUNING:-true}"
export MODEL="${MODEL:-Llama3.2-3B}"
export QUANTIZATION_SCHEME="${QUANTIZATION_SCHEME:-fp8-int4}"

# Set up LOG_DIR
if [[ "$FULL_FINETUNING" == "true" ]]; then
    FINETUNING_TYPE="full"
else
    FINETUNING_TYPE="lora"
fi
LOG_DIR="${BASE_LOG_DIR}/${MODEL}-${QUANTIZATION_SCHEME}-${FINETUNING_TYPE}"
if [[ -n "$RUN_TAG" ]]; then
    LOG_DIR="${LOG_DIR}-${RUN_TAG}"
fi

# Set up OUTPUT_DIR
if [[ "$ENABLE_QAT" == "true" ]]; then
    LOG_FILE="${LOG_DIR}/run_qat_${QUANTIZATION_SCHEME}.log"
    export OUTPUT_DIR="${LOG_DIR}/unsloth_model_${FINETUNING_TYPE}_qat_${QUANTIZATION_SCHEME}_output"
    export QAT_SCHEME="$QUANTIZATION_SCHEME"
else
    LOG_FILE="${LOG_DIR}/run_baseline.log"
    export OUTPUT_DIR="${LOG_DIR}/unsloth_model_${FINETUNING_TYPE}_baseline_output"
fi

# Finetune
if [[ "$SKIP_FINETUNE" != "true" ]]; then
    mkdir -p "$LOG_DIR"
    python finetune.py > "$LOG_FILE" 2>&1
fi

# Eval
export MODEL_DIR="$OUTPUT_DIR"
env -u QUANTIZATION_SCHEME python quantize_and_test_fibonacci.py > "${OUTPUT_DIR}/fib_eval_float.log" 2>&1
python quantize_and_test_fibonacci.py > "${OUTPUT_DIR}/fib_eval_quantized.log" 2>&1
accelerate launch -m lm_eval --model hf --model_args pretrained="${OUTPUT_DIR}" --tasks wikitext --batch_size 2 > "${OUTPUT_DIR}/lm_eval_float.log" 2>&1
accelerate launch -m lm_eval --model hf --model_args pretrained="${OUTPUT_DIR}_quantized2" --tasks wikitext --batch_size 2 > "${OUTPUT_DIR}/lm_eval_quantized.log" 2>&1
