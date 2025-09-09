LOG_DIR="${LOG_DIR:-/home/andrewor/local/logs/unsloth}"
FULL_FINETUNING="${FULL_FINETUNING:-true}"

export QUANTIZATION_SCHEME="${QUANTIZATION_SCHEME:-fp8-int4}"

# Finetune
if [[ "$SKIP_FINETUNE" != "true" ]]; then
    rm -rf "$LOG_DIR/unsloth"*
    export SAVE_OUTPUT_DIR="$LOG_DIR"
    if [[ "$FULL_FINETUNING" == "true" ]]; then
        CUDA_VISIBLE_DEVICES=0 FULL_FINETUNING="true" python finetune.py \
            > "${LOG_DIR}/unsloth_full_baseline.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 FULL_FINETUNING="true" QAT_SCHEME="$QUANTIZATION_SCHEME" python finetune.py \
            > "${LOG_DIR}/unsloth_full_qat_${QUANTIZATION_SCHEME}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=2 FULL_FINETUNING="false" python finetune.py \
            > "${LOG_DIR}/unsloth_lora_baseline.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=3 FULL_FINETUNING="false" QAT_SCHEME="$QUANTIZATION_SCHEME" python finetune.py \
            > "${LOG_DIR}/unsloth_lora_qat_${QUANTIZATION_SCHEME}.log" 2>&1 &
    fi
    wait
fi

# Eval
do_eval() {
    local MODEL_DIR=$1
    export MODEL_DIR
    LOG_DIR="$MODEL_DIR" MODEL="Llama3.1-8B" ./torchtune_quantize_eval.sh
    env -u QUANTIZATION_SCHEME python quantize_and_test_fibonacci.py > "${MODEL_DIR}/fib_eval_float.log" 2>&1
    python quantize_and_test_fibonacci.py > "${MODEL_DIR}/fib_eval_quantized.log" 2>&1
    accelerate launch -m lm_eval --model hf --model_args pretrained="${MODEL_DIR}" --tasks wikitext --batch_size 2 > "${MODEL_DIR}/lm_eval_float.log" 2>&1
    accelerate launch -m lm_eval --model hf --model_args pretrained="${MODEL_DIR}_quantized2" --tasks wikitext --batch_size 2 > "${MODEL_DIR}/lm_eval_quantized.log" 2>&1
}

if [[ "$FULL_FINETUNING" == "true" ]]; then
    CUDA_VISIBLE_DEVICES=0 do_eval "${LOG_DIR}/unsloth_model_full_baseline_output" &
    CUDA_VISIBLE_DEVICES=1 do_eval "${LOG_DIR}/unsloth_model_full_qat_${QUANTIZATION_SCHEME}_output" &
else
    CUDA_VISIBLE_DEVICES=0 do_eval "${LOG_DIR}/unsloth_model_lora_baseline_output" &
    CUDA_VISIBLE_DEVICES=1 do_eval "${LOG_DIR}/unsloth_model_lora_qat_${QUANTIZATION_SCHEME}_output" &
fi
wait
