export QUANTIZATION_SCHEME="fp8-int4"
export MODEL="Qwen3-8B"

FULL_FINETUNING=false BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=2e-4 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr2e-4-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr2e-4-1epoch

FULL_FINETUNING=false BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=1e-4 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr1e-4-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr1e-4-1epoch

FULL_FINETUNING=false BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=4e-5 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr4e-5-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr4e-5-1epoch

FULL_FINETUNING=false BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=2e-5 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr2e-5-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs16-lr2e-5-1epoch

FULL_FINETUNING=false BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=2e-4 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr2e-4-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr2e-4-1epoch

FULL_FINETUNING=false BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=1e-4 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr1e-4-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr1e-4-1epoch

FULL_FINETUNING=false BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=4e-5 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr4e-5-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr4e-5-1epoch

FULL_FINETUNING=false BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=-1 LEARNING_RATE=2e-5 ./super_run_unsloth.sh
mkdir -p /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr2e-5-1epoch
mv /home/andrewor/local/logs/unsloth/unsloth* /home/andrewor/local/logs/unsloth/saved-9-16-lora-bs8-lr2e-5-1epoch
