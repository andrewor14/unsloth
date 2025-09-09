# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

GROUP_SIZE="${GROUP_SIZE:-32}"
QUANTIZATION_SCHEME="${QUANTIZATION_SCHEME:-fp8-int4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
FIRST_GPU="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | head -n 1)"


if [[ -z "$LOG_DIR" ]]; then
    echo "LOG_DIR must be specified"
    exit 1
fi


# Quantizer
if [[ "$QUANTIZATION_SCHEME" == "nvfp4" ]]; then
    QUANTIZER="torchtune.training.quantization.NVFP4Quantizer"
    GROUP_SIZE="16"
elif [[ "$QUANTIZATION_SCHEME" == "fp8-int4" ]]; then
    QUANTIZER="torchtune.training.quantization.Float8ActivationInt4WeightQuantizer"
elif [[ "$QUANTIZATION_SCHEME" == "int8-int4" ]]; then
    QUANTIZER="torchtune.training.quantization.Int8DynActInt4WeightQuantizer"
    QUANTIZATION_SCHEME="8da4w"
elif [[ "$QUANTIZATION_SCHEME" == "int4" ]]; then
    QUANTIZER="torchtune.training.quantization.Int4WeightOnlyQuantizer"
    QUANTIZATION_SCHEME="4w"
else
    echo "Unknown quantization scheme $QUANTIZATION_SCHEME"
    exit 1
fi


# Model type
if [[ -z "$MODEL" ]]; then
    MODEL="Llama3-8B"
fi
if [[ "$MODEL" == "Llama3.2-3B" ]]; then
    MODEL_COMPONENT="torchtune.models.llama3_2.llama3_2_3b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00002-${QUANTIZATION_SCHEME}.ckpt"
    MODEL_TYPE="LLAMA3"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    TOKENIZER_PATH="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    TOKENIZER_MERGES_FILE="null"
elif [[ "$MODEL" == "Llama3.1-8B" ]]; then
    MODEL_COMPONENT="torchtune.models.llama3_1.llama3_1_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00004-${QUANTIZATION_SCHEME}.ckpt"
    MODEL_TYPE="LLAMA3"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    TOKENIZER_PATH="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    TOKENIZER_MERGES_FILE="null"
elif [[ "$MODEL" == "Llama3-8B" ]]; then
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00004-${QUANTIZATION_SCHEME}.ckpt"
    MODEL_TYPE="LLAMA3"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    TOKENIZER_PATH="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    TOKENIZER_MERGES_FILE="null"
elif [[ "$MODEL" == "Qwen3-1.7B" ]]; then
    MODEL_COMPONENT="torchtune.models.qwen3.qwen3_1_7b_instruct"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00001.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00001-${QUANTIZATION_SCHEME}.ckpt"
    MODEL_TYPE="QWEN3"
    TOKENIZER_COMPONENT="torchtune.models.qwen3.qwen3_tokenizer"
    TOKENIZER_PATH="/tmp/Qwen3-1.7B/vocab.json"
    TOKENIZER_MERGES_FILE="/tmp/Qwen3-1.7B/merges.txt"
else
    echo "Unknown model $MODEL"
    exit 1
fi


# Eval
if [[ "$SKIP_EVAL" != "true" ]]; then
    echo "Quantizing, logging to ${LOG_DIR}/quantize.log"
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run quantize --config quantization \
        model._component_="$MODEL_COMPONENT" \
        checkpointer._component_="$CHECKPOINTER" \
        checkpointer.checkpoint_dir="$LOG_DIR" \
        checkpointer.output_dir="${LOG_DIR}_quantized" \
        checkpointer.checkpoint_files="$CHECKPOINT_FILES" \
        checkpointer.model_type="$MODEL_TYPE" \
        quantizer._component_="$QUANTIZER" \
        quantizer.groupsize="$GROUP_SIZE" \
        > "${LOG_DIR}/quantize.log" 2>&1
    cp "${LOG_DIR}_quantized/${QUANTIZED_CHECKPOINT_FILE}" "$LOG_DIR"

    if [[ "$MODEL_TYPE" == "LLAMA3" ]]; then
        echo "Evaluating wikitext (float), logging to ${LOG_DIR}/eval_wikitext_float.log"
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
            batch_size=1 \
            tasks=[wikitext] \
            model._component_="$MODEL_COMPONENT" \
            checkpointer._component_="$CHECKPOINTER" \
            checkpointer.checkpoint_dir="$LOG_DIR" \
            checkpointer.output_dir="${LOG_DIR}_quantized" \
            checkpointer.checkpoint_files="$CHECKPOINT_FILES" \
            checkpointer.model_type="$MODEL_TYPE" \
            tokenizer._component_="$TOKENIZER_COMPONENT" \
            tokenizer.path="$TOKENIZER_PATH" \
            > "${LOG_DIR}/eval_wikitext_float.log" 2>&1 &

        echo "Evaluating wikitext (quantized), logging to ${LOG_DIR}/eval_wikitext_quantized.log"
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
            batch_size=1 \
            tasks=[wikitext] \
            model._component_="$MODEL_COMPONENT" \
            checkpointer._component_="torchtune.training.FullModelTorchTuneCheckpointer" \
            checkpointer.checkpoint_dir="$LOG_DIR" \
            checkpointer.output_dir="${LOG_DIR}_quantized" \
            checkpointer.checkpoint_files="[$QUANTIZED_CHECKPOINT_FILE]" \
            checkpointer.model_type="$MODEL_TYPE" \
            tokenizer._component_="$TOKENIZER_COMPONENT" \
            tokenizer.path="$TOKENIZER_PATH" \
            quantizer._component_="$QUANTIZER" \
            quantizer.groupsize="$GROUP_SIZE" \
            > "${LOG_DIR}/eval_wikitext_quantized.log" 2>&1 &
    elif [[ "$MODEL_TYPE" == "QWEN3" ]]; then
        echo "Evaluating wikitext (float), logging to ${LOG_DIR}/eval_wikitext_float.log"
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
            batch_size=1 \
            tasks=[wikitext] \
            model._component_="$MODEL_COMPONENT" \
            checkpointer._component_="$CHECKPOINTER" \
            checkpointer.checkpoint_dir="$LOG_DIR" \
            checkpointer.output_dir="${LOG_DIR}_quantized" \
            checkpointer.checkpoint_files="$CHECKPOINT_FILES" \
            checkpointer.model_type="$MODEL_TYPE" \
            tokenizer._component_="$TOKENIZER_COMPONENT" \
            tokenizer.path="$TOKENIZER_PATH" \
            tokenizer.merges_file="$TOKENIZER_MERGES_FILE" \
            > "${LOG_DIR}/eval_wikitext_float.log" 2>&1 &

        echo "Evaluating wikitext (quantized), logging to ${LOG_DIR}/eval_wikitext_quantized.log"
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
            batch_size=1 \
            tasks=[wikitext] \
            model._component_="$MODEL_COMPONENT" \
            checkpointer._component_="torchtune.training.FullModelTorchTuneCheckpointer" \
            checkpointer.checkpoint_dir="$LOG_DIR" \
            checkpointer.output_dir="${LOG_DIR}_quantized" \
            checkpointer.checkpoint_files="[$QUANTIZED_CHECKPOINT_FILE]" \
            checkpointer.model_type="$MODEL_TYPE" \
            tokenizer._component_="$TOKENIZER_COMPONENT" \
            tokenizer.path="$TOKENIZER_PATH" \
            tokenizer.merges_file="$TOKENIZER_MERGES_FILE" \
            quantizer._component_="$QUANTIZER" \
            quantizer.groupsize="$GROUP_SIZE" \
            > "${LOG_DIR}/eval_wikitext_quantized.log" 2>&1 &
    fi
    wait
fi
