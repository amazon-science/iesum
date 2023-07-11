# Copyright <first-edit-year> Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import json
from datasets import load_metric
import os

def compute_rouge(data_list, output_dir):
    predictions = [item["generated_summary"] for item in data_list]
    references = [item["gold_summary"] for item in data_list]

    scorer = load_metric("rouge")
    s = scorer.compute(predictions=predictions, references=references, use_stemmer=True)

    results = {
        "rouge-1": {
            "p": s["rouge1"][0].precision,
            "r": s["rouge1"][0].recall,
            "f": s["rouge1"][0].fmeasure,
        },
        "rouge-2": {
            "p": s["rouge2"][0].precision,
            "r": s["rouge2"][0].recall,
            "f": s["rouge2"][0].fmeasure,
        },
        "rouge-L": {
            "p": s["rougeL"][0].precision,
            "r": s["rougeL"][0].recall,
            "f": s["rougeL"][0].fmeasure,
        },
        "rouge-Lsum": {
            "p": s["rougeLsum"][0].precision,
            "r": s["rougeLsum"][0].recall,
            "f": s["rougeLsum"][0].fmeasure,
        }
    }

    with open(os.path.join(output_dir, "rouge.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4))