# Example command:
# find /home/andrewor/local/logs/unsloth/saved-9-19 -maxdepth 1 -type d | sed 1d | xargs python parse_it.py

import os
import re
import sys

if len(sys.argv) == 1:
    print("Usage: python parse_it.py [log_dir1] [log_dir2] ...")
    sys.exit(1)
log_dirs = sys.argv[1:]

PREFIX_TO_CUT = os.getenv("PREFIX_TO_CUT", "")

QAT_NAME = "unsloth_model_lora_qat_int4_output"
BASELINE_NAME = "unsloth_model_lora_baseline_output"

def extract_wikitext_perplexity(log_file: str) -> float:
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|word_perplexity\|↓  \|([\\d. ]*)\|.*", l)
            if m is not None:
                return float(m.groups()[0])
    return -1

def extract_bbh_accuracy(log_file: str) -> float:
    # |bbh     |      3|get-answer    |      |exact_match|↑  |0.7068|±  |0.0050|
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match("\|bbh.*\|([\\d. ]*)\|±.*", l)
            if m is not None:
                return float(m.groups()[0])
    return -1

def extract_gpqa_accuracy(log_file: str) -> float:
    # |gpqa_diamond_n_shot  | 2|none |  0|acc    |↑  |0.3687|±  |0.0344|
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match("\|gpqa_diamond_n_shot.*\|([\\d. ]*)\|±.*", l)
            if m is not None:
                return float(m.groups()[0])
    return -1

def extract_mmlu_pro_accuracy(log_file: str) -> float:
    # |mmlu_pro|      2|custom-extract|      |exact_match|↑  |0.4120|±  |0.0044|
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match("\|mmlu_pro.*\|([\\d. ]*)\|±.*", l)
            if m is not None:
                return float(m.groups()[0])
    return -1

task_to_parse_fn = {
    "wikitext": extract_wikitext_perplexity,
    "bbh": extract_bbh_accuracy,
    "gpqa": extract_gpqa_accuracy,
    "mmlu_pro": extract_mmlu_pro_accuracy,
}

for log_dir in log_dirs:
    # extract eval data
    all_data = {}
    try:
        for experiment_name in [QAT_NAME, BASELINE_NAME]:
            experiment_dir = os.path.join(log_dir, experiment_name)
            all_data[experiment_name] = {}
            float_eval_file = os.path.join(experiment_dir, "new_lm_eval_float.log")
            quantized_eval_file = os.path.join(experiment_dir, "new_lm_eval_quantized.log")
            for task, parse_fn in task_to_parse_fn.items():
                if experiment_name != QAT_NAME:
                    all_data[experiment_name][f"{task}_float"] = parse_fn(float_eval_file)
                all_data[experiment_name][f"{task}_quantized"] = parse_fn(quantized_eval_file)
    except FileNotFoundError:
        print(f"Skipping log directory {log_dir}")
        continue
    
    # print data in a nice format
    for task in task_to_parse_fn.keys():
        baseline_float_value = all_data[BASELINE_NAME][task + "_float"]
        baseline_quantized_value = all_data[BASELINE_NAME][task + "_quantized"]
        qat_quantized_value = all_data[QAT_NAME][task + "_quantized"]
        if baseline_float_value == -1 or baseline_quantized_value == -1:
            print(
                f"Warning: baseline float value for task {task} was {baseline_float_value}, "
                f"quantized value was {baseline_quantized_value}"
            )
            continue
        recovered = (qat_quantized_value - baseline_quantized_value) / (baseline_float_value - baseline_quantized_value) * 100
        print(
            "%s: %s (baseline) -> %s (quant) -> %s (qat), recovered %.3f%% (%s)" % (
                log_dir.split("/")[-1].replace(PREFIX_TO_CUT, ""),
                baseline_float_value,
                baseline_quantized_value,
                qat_quantized_value,
                recovered,
                task,
            )
        )
