# Example command:
# find /home/andrewor/local/logs/unsloth/saved-9-19 -maxdepth 1 -type d | sed 1d | xargs python parse_it.py

import os
import re
import sys

if len(sys.argv) == 1:
    print("Usage: python parse_it.py [log_dir1] [log_dir2] ...")
    sys.exit(1)
log_dirs = sys.argv[1:]

QAT_NAME = "unsloth_model_lora_qat_int4_output"
BASELINE_NAME = "unsloth_model_lora_baseline_output"

def extract_wikitext_perplexity(log_file: str) -> float:
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|word_perplexity\|↓  \|([\\d.]*)\|.*", l)
            if m is not None:
                return float(m.groups()[0])
    raise ValueError("Did not find wikitext perplexity in %s" % log_file)

for log_dir in log_dirs:
    # extract eval data
    all_data = {}
    for experiment_name in [QAT_NAME, BASELINE_NAME]:
        experiment_dir = os.path.join(log_dir, experiment_name)
        all_data[experiment_name] = {}
        float_eval_file = os.path.join(experiment_dir, "lm_eval_float.log")
        quantized_eval_file = os.path.join(experiment_dir, "lm_eval_quantized.log")
        all_data[experiment_name]["wikitext_word_perplexity_float"] = extract_wikitext_perplexity(float_eval_file)
        all_data[experiment_name]["wikitext_word_perplexity_quantized"] = extract_wikitext_perplexity(quantized_eval_file)
    
    # print data in a nice format
    metric = "wikitext_word_perplexity"
    baseline_float_value = all_data[BASELINE_NAME][metric + "_float"]
    baseline_quantized_value = all_data[BASELINE_NAME][metric + "_quantized"]
    qat_quantized_value = all_data[QAT_NAME][metric + "_quantized"]
    recovered = (qat_quantized_value - baseline_quantized_value) / (baseline_float_value - baseline_quantized_value) * 100
    print(
        "%s: baseline %.3f -> %.3f (quant) -> %.3f (qat), recovered %.3f%%" % (
            log_dir.split("/")[-1],
            baseline_float_value,
            baseline_quantized_value,
            qat_quantized_value,
            recovered,
        )
    )
