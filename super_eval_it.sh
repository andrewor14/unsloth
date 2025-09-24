# Others to eval:
# /home/andrewor/local/logs/unsloth/saved-9-22-lora/Qwen3-4B-Instruct-int4-lora-lr2e-5
# /home/andrewor/local/logs/unsloth/saved-9-22-lora/Qwen3-8B-int4-lora-lr2e-5
# /home/andrewor/local/logs/unsloth/saved-9-22-lora/Llama3.2-3B-int4-lora-lr2e-5
# /home/andrewor/local/logs/unsloth/saved-9-22-lora/Llama3.2-1B-int4-lora-lr2e-5

LM_EVAL_TASKS="wikitext,bbh,mmlu_pro,gpqa" # this takes 24 hours per run!

for LOG_DIR in "/home/andrewor/local/logs/unsloth/saved-9-22-lora/Gemma3-12B-int4-lora-lr2e-5" "/home/andrewor/local/logs/unsloth/saved-9-22-lora/Gemma3-4B-int4-lora-lr2e-5"; do
    BASELINE_DIR="${LOG_DIR}/unsloth_model_lora_baseline_output"
    QAT_DIR="${LOG_DIR}/unsloth_model_lora_qat_int4_output"
    echo "Running lm_eval on ${BASELINE_DIR}"
    echo "Running lm_eval on ${BASELINE_DIR}_quantized2"
    echo "Running lm_eval on ${QAT_DIR}_quantized2"
    CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf --model_args pretrained="${BASELINE_DIR}" --tasks "$LM_EVAL_TASKS" --batch_size auto > "${BASELINE_DIR}/new_lm_eval_float.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 accelerate launch -m lm_eval --model hf --model_args pretrained="${BASELINE_DIR}_quantized2" --tasks "$LM_EVAL_TASKS" --batch_size auto > "${BASELINE_DIR}/new_lm_eval_quantized.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 accelerate launch -m lm_eval --model hf --model_args pretrained="${QAT_DIR}_quantized2" --tasks "$LM_EVAL_TASKS" --batch_size auto > "${QAT_DIR}/new_lm_eval_quantized.log" 2>&1 &
    wait
done
