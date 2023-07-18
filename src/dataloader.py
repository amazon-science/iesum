# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/


from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import torch
from random import shuffle
import random
import os
import re
import sys
import json
import tqdm
import math
import dgl


def transform_offsets(start, end, offsets_list):
    curr_list = offsets_list[1:-1].copy()
    length = len(offsets_list)
    curr_list.append((math.inf, math.inf))

    start_idx, end_idx = 0, 1

    for i in range(length - 1):
        if start > curr_list[i][0] and start <= curr_list[i+1][0]:
            start_idx = i+1
        if end > curr_list[i][0] and end <= curr_list[i+1][0]:
            end_idx = i+1

    return start_idx, end_idx


def generate_mention_mask(start, end, total_len):
    # print(start, end, total_len)
    weight = 1 / (end - start)
    output = []
    for i in range(total_len):
        if i >= start and i < end:
            output.append(weight)
        else:
            output.append(0.0)
    return output


class SummarizationDataset(Dataset):
    def __init__(
        self,
        args,
        dataset_dir,
        dataset_name,
        join_method,
        tokenizer,
        max_input_len,
        max_output_len,
        mask_num=5,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="train",
    ):
        # self.hf_dataset = hf_dataset
        self.dataset_name = dataset_name
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        if join_method == "concat_start_wdoc_global":
            if self.tokenizer.name_or_path == 'allenai/PRIMERA':
                self.docsep_token_id = self.tokenizer.vocab_size
            else:
                self.docsep_token_id = self.tokenizer.vocab_size - 1
        # print(self.docsep_token_id)
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        self.dataset_type = dataset_type

        # if args.data_load_method == "json":
        #     self.load_data_from_json(dataset_dir)
        #     names = dataset_dir.split('/')
        #     self.save_data_to_pkl(os.path.join("/".join(names[:-1]), names[-1].split(".")[0]+".pkl"))
        # elif args.data_load_method == "pkl":
        #     print("load cached pkl dataset from: " + dataset_dir)
        #     self.load_data_from_pkl(dataset_dir)

        if dataset_dir.endswith(".pkl"):
            print("load cached pkl dataset from: " + dataset_dir)
            self.load_data_from_pkl(dataset_dir)
        else:
            self.load_data_from_json(dataset_dir)
            names = dataset_dir.split('/')
            self.save_data_to_pkl(os.path.join("/".join(names[:-1]), names[-1].split(".")[0]+ "_" + str(max_output_len) + ".pkl"))
        
        if num_data != -1 and num_data < len(list(self.examples)):
            random.seed(rand_seed)
            self.examples = random.sample(list(self.examples), num_data)
        

    def load_data_from_json(self, json_dir):
        with open(json_dir, 'r', encoding="utf-8") as f:
            items = json.loads(f.read())

        examples = []
        progress = tqdm.tqdm(total=len(items), ncols=120, desc='Processing Data: '+json_dir)
        for i,item in enumerate(items):
            examples.append(self.process_item(item, i))
            progress.update(1)

        progress.close()
        self.examples = examples

        # names = self.data_dir.split('/')
        # torch.save(examples, os.path.join("/".join(names[:-1]), names[-1].split(".")[0]+".pkl"))
    
    def save_data_to_pkl(self, pkl_dir):
        torch.save(self.examples, pkl_dir)
    
    def load_data_from_pkl(self, pkl_dir):
        self.examples = torch.load(pkl_dir)
    
    def process_item(self, item, idx):
        # item: a data entry 
        all_docs = item["articles"]
        tgt = item["summary"]

        # read edges to build graph
        edges = []
        for start, ends in item["edges"].items():
            for end in ends:
                edges.append([int(start), int(end)])
        edges = torch.LongTensor(edges).T 
        graph = dgl.graph((edges[0], edges[1]))

        # We first collect all spans corresponding with their nodes/types in each document.
        node_num, mention_num = 0, 0
        spans = {i:[] for i in range(len(all_docs))}
        for str_idx, node in item["nodes"].items():
            if int(str_idx) > node_num:
                node_num = int(str_idx)
            node_type = node["type"]
            for span in node["spans"]:
                doc_idx, start, end = span
                spans[doc_idx].append((str_idx, start, end, node_type))
                mention_num += 1

        node_num += 1
        input_ids, labels, initial_mask = [], [], []
        mentions= [] # (front_len, [list]) to be padded afterwards
        node_mask = [[0.0 for q in range(mention_num)] for p in range(node_num)].copy()

        mention_idx = 0
    
        for i,doc in enumerate(all_docs):
            tokens = self.tokenizer(doc, 
                        return_offsets_mapping=True, 
                        truncation=True,
                        max_length=(self.max_input_len) // len(all_docs)
                    )
            current_len = len(input_ids)
            input_ids.extend(tokens["input_ids"][1:-1])

            label_segs = [0 for _ in range(len(tokens["input_ids"][1:-1]))]

            mapping = tokens["offset_mapping"]
            text_spans = spans[i]
            for span in text_spans:
                # span: (node_idx, start, end, type)
                node_idx, start, end, node_type = span
                start_offset, end_offset = transform_offsets(start, end, mapping)

                for j in range(start_offset, end_offset):
                    if node_type == "entity":
                        if j == start_offset:
                            label_segs[j] = 1
                        else:
                            label_segs[j] = 2
                    if node_type == "event":
                        if j == start_offset:
                            label_segs[j] = 3
                        else:
                            label_segs[j] = 4
                
                mention_mask_list = [0.0 for _ in range(current_len)] + generate_mention_mask(start_offset, end_offset, len(tokens["input_ids"][1:-1]))
                mentions.append(mention_mask_list.copy())
                node_mask[int(node_idx)][mention_idx] = 1.0
                mention_idx += 1
            
            labels.extend(label_segs)

            if i != len(all_docs) - 1:
                input_ids.append(self.docsep_token_id)
                labels.append(0)

        input_ids = ([self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id])
        labels = ([0] + labels + [0])

        # pad mention_mask here:
        output_mention_masks = []
        for mask_list in mentions:
            output_mention_masks.append([0.0] + mask_list.copy() + [0.0 for _ in range(len(input_ids) - 1 - len(mask_list))].copy())

        mention_indicators = [1.0 for _ in range(mention_num)]
        node_indicators = [1.0 for _ in range(node_num)]

        output_ids = self.tokenizer.encode(
            tgt, truncation=True, max_length=self.max_output_len
        )
        # print(sum(output_mention_masks[0]))
        if self.dataset_type == "train":
            return idx, torch.tensor(input_ids), torch.tensor(output_ids), torch.tensor(labels), output_mention_masks, mention_indicators, node_mask, node_indicators, graph
        else:
            return idx, torch.tensor(input_ids), torch.tensor(output_ids), torch.tensor(labels), output_mention_masks, mention_indicators, node_mask, node_indicators, graph, all_docs, tgt


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    if batch[0][1][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0][1][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        raise AssertionError
    train = True
    if len(batch[0]) == 11:
        train = False
        all_docs = [item[8] for item in batch]
        tgt = [item[9] for item in batch]
        idxs = [item[0] for item in batch]
        new_batch = [item[1:4] for item in batch]
    else:
        idxs = [item[0] for item in batch]
        new_batch = [item[1:4] for item in batch]

    input_ids, output_ids, labels = list(zip(*new_batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    
    # pad mention mask
    batch_mention_masks = [item[4] for item in batch]
    batch_mention_indicators = [item[5] for item in batch]
    batch_node_masks = [item[6] for item in batch]
    batch_node_indicators = [item[7] for item in batch]

    new_batch_mention_masks, new_batch_node_masks = [], []

    max_seq_len = 0
    max_mention_num = max([len(mask) for mask in batch_mention_masks])
    for mask in batch_mention_masks:
        for m in mask:
            if len(m) > max_seq_len:
                max_seq_len = len(m)
    max_node_num = max([len(mask) for mask in batch_node_masks])
    # print(input_ids.shape)
    # print(max_seq_len)

    mention_indicators = torch.tensor([mask + [0.0 for _ in range(max_mention_num - len(mask))] for mask in batch_mention_indicators])
    node_indicators = torch.tensor([mask + [0.0 for _ in range(max_node_num - len(mask))] for mask in batch_node_indicators])

    for i in range(len(batch)):
        mention_mask = batch_mention_masks[i]
        new_mention = []
        for mask in mention_mask:
            new_mention.append((mask + [0.0 for x in range(max_seq_len - len(mask))]).copy())
        new_mask = new_mention + [[0.0 for z in range(max_seq_len)] for y in range(max_mention_num - len(mention_mask))]
        new_batch_mention_masks.append(new_mask.copy())

    for i in range(len(batch)):
        node_mask = batch_node_masks[i]
        new_node = [mask + [0.0 for _ in range(max_mention_num - len(mask))] for mask in node_mask].copy()
        new_mask = new_node + [[0.0 for z in range(max_mention_num)] for _ in range(max_node_num - len(node_mask))]
        new_batch_node_masks.append(new_mask.copy())
    
    lengths = []
    for mask in new_batch_mention_masks:
        for m in mask:
            lengths.append(len(m))
    
    new_batch_mention_masks = torch.tensor(new_batch_mention_masks)
    new_batch_node_masks = torch.tensor(new_batch_node_masks)

    graphs = [item[8] for item in batch]

    idxs = torch.LongTensor(idxs)
    if train:
        return {
            "idxs": idxs,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "labels": labels,
            "mention_select": new_batch_mention_masks,
            "mention_mask": mention_indicators,
            "node_select": new_batch_node_masks,
            "node_mask": node_indicators,
            "graphs": graphs
        }
    else:
        return {
            "idxs": idxs,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "labels": labels,
            "mention_select": new_batch_mention_masks,
            "mention_mask": mention_indicators,
            "node_select": new_batch_node_masks,
            "node_mask": node_indicators,
            "graphs": graphs,
            "all_docs": all_docs,
            "tgt": tgt
        }


def get_dataloader_summ(args, dataset_name, tokenizer, split_name, num_workers, is_train):
    data_path = os.path.join(args.data_path, dataset_name)
    # print(data_path)
    if split_name == "train":
        
        if args.setting == "baseline":
            if os.path.exists(os.path.join(data_path, "train_" + str(args.max_length_tgt) + ".pkl")):
                data_file_path = os.path.join(data_path, "train_" + str(args.max_length_tgt) + ".pkl")
            else:
                data_file_path = os.path.join(data_path, "train.json")
        else:
            if os.path.exists(os.path.join(data_path, "train_with_ie_" + str(args.max_length_tgt) + ".pkl")):
                data_file_path = os.path.join(data_path, "train_with_ie_" + str(args.max_length_tgt) + ".pkl")
            else:
                data_file_path = os.path.join(data_path, "train_with_ie.json")


        dataset = SummarizationDataset(
            args=args,
            dataset_dir=data_file_path,
            dataset_name=args.dataset_name,
            join_method=args.join_method,
            tokenizer=tokenizer,
            max_input_len=args.max_length_input,
            max_output_len=args.max_length_tgt,
            mask_num=args.mask_num,
            num_data=args.num_train_data,
            rand_seed=args.rand_seed,
            is_test=(split_name == "test"),
            dataset_type=split_name,
        )

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True
        )

    elif split_name == "validation":
        if args.setting == "baseline":
            if os.path.exists(os.path.join(data_path, "val_" + str(args.max_length_tgt) + ".pkl")):
                data_file_path = os.path.join(data_path, "val_" + str(args.max_length_tgt) + ".pkl")
            else:
                data_file_path = os.path.join(data_path, "val.json")
        else:
            if os.path.exists(os.path.join(data_path, "val_with_ie_" + str(args.max_length_tgt) + ".pkl")):
                data_file_path = os.path.join(data_path, "val_with_ie_" + str(args.max_length_tgt) + ".pkl")
            else:
                data_file_path = os.path.join(data_path, "val_with_ie.json")

        dataset = SummarizationDataset(
            args=args,
            dataset_dir=data_file_path,
            dataset_name=args.dataset_name,
            join_method=args.join_method,
            tokenizer=tokenizer,
            max_input_len=args.max_length_input,
            max_output_len=args.max_length_tgt,
            mask_num=args.mask_num,
            num_data=args.num_valid_data,
            rand_seed=args.rand_seed,
            is_test=(split_name == "test"),
            dataset_type=split_name,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

    else:

        if args.setting == "baseline":
            if os.path.exists(os.path.join(data_path, "test_" + str(args.max_length_tgt) + ".pkl")):
                data_file_path = os.path.join(data_path, "test_" + str(args.max_length_tgt) + ".pkl")
            else:
                data_file_path = os.path.join(data_path, "test.json")
        else:
            if os.path.exists(os.path.join(data_path, "test_with_ie_" + str(args.max_length_tgt) + ".pkl")):
                data_file_path = os.path.join(data_path, "test_with_ie_" + str(args.max_length_tgt) + ".pkl")
            else:
                data_file_path = os.path.join(data_path, "test_with_ie.json")

        dataset = SummarizationDataset(
            args=args,
            dataset_dir=data_file_path,
            dataset_name=args.dataset_name,
            join_method=args.join_method,
            tokenizer=tokenizer,
            max_input_len=args.max_length_input,
            max_output_len=args.max_length_tgt,
            mask_num=args.mask_num,
            num_data=args.num_test_data,
            rand_seed=args.rand_seed,
            is_test=(split_name == "test"),
            dataset_type=split_name,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )


if __name__ == "__main__":
    # l = [(0, 0), (0, 1), (2, 6), (7, 10), (10, 11), (0, 0)]
    # print(transform_offsets(2,6,l))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="debug", type=str, help="dataset name: multi_news, wcep, wcep10")
    parser.add_argument("--tgt_length", default=256, type=int, help="length of targeted summary")
    parser.add_argument("--max_length_input", default=4096, type=int, help="length of input articles")
    parser.add_argument("--max_length_tgt", default=256, type=int, help="length of targeted summary")
    parser.add_argument("--data_load_method", default="json", type=str, help="json/pkl")
    parser.add_argument("--split", default="test", type=str, help="test/train/val")
    parser.add_argument("--setting", default="baseline", type=str, help="test/train/val")
    parser.add_argument("--num_workers", default=8, type=int, help="test/train/val")
    parser.add_argument("--mask_num", default=0, type=int, help="test/train/val")
    parser.add_argument("--num_train_data", default=-1, type=int, help="test/train/val")
    parser.add_argument("--num_valid_data", default=-1, type=int, help="test/train/val")
    parser.add_argument("--num_test_data", default=-1, type=int, help="test/train/val")
    parser.add_argument("--data_path", default="./data/", type=str, help="test/train/val")
    parser.add_argument("--rand_seed", default=61801, type=int, help="test/train/val")
    parser.add_argument("--batch_size", default=3, type=int, help="test/train/val")
    parser.add_argument("--join_method", default="concat_start_wdoc_global", type=str, help="concat_start_wdoc_global")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    # r = AutoTokenizer.from_pretrained("allenai/PRIMERA")
    r = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    print(r)

    d = get_dataloader_summ(args, args.dataset_name, r, "train", args.num_workers, True)
    for l in d:
        pass
    # get_dataloader_summ(args, args.dataset_name, r, "validation", args.num_workers, True)
    # get_dataloader_summ(args, args.dataset_name, r, "test", args.num_workers, True)
    
    # batch_node_masks = [[[1,2,3], [4,5,6]], [[1,2,3,4], [5,6,7,8], [9,10,11,12]]]
    # new_batch_node_masks = []

    # for i in range(2):
    #     node_mask = batch_node_masks[i]
    #     new_node = [mask + [0.0 for _ in range(4 - len(mask))] for mask in node_mask]
    #     new_mask = new_node + [[0.0 for z in range(4)] for _ in range(3 - len(node_mask))]
    #     new_batch_node_masks.append(new_mask.copy())
    
    # print(new_batch_node_masks)


