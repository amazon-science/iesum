# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import json
import argparse
import os

from copy import deepcopy

from metrics.rouge import compute_rouge
from metrics.mint import compute_mint
from metrics.entity import compute_entity_precision
from metrics.factcc import compute_factcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory of test dumps.")
    parser.add_argument("--output_dir", type=str, help="Directory of output_dir")
    parser.add_argument("--metric", type=str, help="entity, mint, rouge, factcc")
    args = parser.parse_args()

    # read data
    data_list = []
    file_list = os.listdir(args.data_dir)

    for filename in file_list:
        if filename.endswith(".json"):
            with open(os.path.join(args.data_dir, filename), 'r', encoding='utf-8') as f:
                data_dict = json.loads(f.read())
                data_list.append(deepcopy(data_dict))
    

    if args.metric == "rouge":
        compute_rouge(data_list, args.output_dir)
    if args.metric == "entity":
        compute_entity_precision(data_list, args.output_dir)
    if args.metric == "mint":
        compute_mint(data_list, args.output_dir)
    if args.metric == "factcc":
        compute_factcc(data_list, args.output_dir)

