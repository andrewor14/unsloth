BASE_LOG_DIR="/home/andrewor/local/logs/unsloth"

export QUANTIZATION_SCHEME="int4"
export FULL_FINETUNING="false"
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_STEPS=-1
export DATASET="mlabonne/FineTome-100k"

# No fine-tune baseline, just evals
#CUDA_VISIBLE_DEVICES=0 MAX_STEPS=0 RUN_TAG="no_finetuning" MODEL="Gemma3-12B" ./super_run_unsloth.sh &
#CUDA_VISIBLE_DEVICES=1 MAX_STEPS=0 RUN_TAG="no_finetuning" MODEL="Qwen3-8B" ./super_run_unsloth.sh &
#wait

# cais/mmlu

export MODEL="Gemma3-12B"
export BATCH_SIZE=16
export MAX_STEPS=100
export DATASET="cais/mmlu"
CUDA_VISIBLE_DEVICES=0 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=2 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=3 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
wait

export MODEL="Qwen3-8B"
export BATCH_SIZE=16
export MAX_STEPS=100
export DATASET="cais/mmlu"
CUDA_VISIBLE_DEVICES=0 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=2 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=3 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
wait


# SlimOrca

export MODEL="Gemma3-12B"
export BATCH_SIZE=16
export MAX_STEPS=500
export DATASET="Open-Orca/SlimOrca"
CUDA_VISIBLE_DEVICES=0 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=2 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=3 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
wait

export MODEL="Qwen3-8B"
export BATCH_SIZE=16
export MAX_STEPS=500
export DATASET="Open-Orca/SlimOrca"
CUDA_VISIBLE_DEVICES=0 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=2 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=3 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
wait


# Open-Platypus

export MODEL="Gemma3-12B"
export BATCH_SIZE=16
export MAX_STEPS=500
export DATASET="garage-bAInd/Open-Platypus"
CUDA_VISIBLE_DEVICES=0 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=2 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=3 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
wait

export MODEL="Qwen3-8B"
export BATCH_SIZE=16
export MAX_STEPS=500
export DATASET="garage-bAInd/Open-Platypus"
CUDA_VISIBLE_DEVICES=0 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=1 LEARNING_RATE=2e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=2 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="true" ./super_run_unsloth.sh &
CUDA_VISIBLE_DEVICES=3 LEARNING_RATE=1e-4 RUN_TAG="${DATASET/\//-}-lr${LEARNING_RATE}" ENABLE_QAT="false" ./super_run_unsloth.sh &
wait
