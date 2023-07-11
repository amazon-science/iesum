# Copyright <first-edit-year> Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

from numpy import NaN
import torch
import os
import json
import sys
import argparse
from transformers import Adafactor
from tqdm import tqdm
from copy import deepcopy

import pandas as pd
import torch.nn as nn
import pdb
import dgl
from datasets import load_dataset, load_metric
import json
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LEDTokenizer,
    LEDForConditionalGeneration,
    LEDConfig,
    LEDModel
)

from gnn import GAT
from transformers.modeling_outputs import ModelOutput
from dataloader import get_dataloader_summ
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pathlib import Path

from typing import List, Optional, Tuple, Union


class LEDSeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


class GraphLED(LEDForConditionalGeneration):
    
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        # Initialize weights and apply final processing
        self.gat = GAT(config.d_model, config.d_model)
        self.ie_linear = nn.Linear(config.d_model, 5)

        self.token_linear = nn.Linear(config.d_model, 256)
        self.graph_linear = nn.Linear(config.d_model, 256)

        self.cosine = nn.CosineSimilarity(dim=2)
        self.post_init()
    
    def get_node_embeddings(self, logits, mention_select, mention_mask, node_select, node_mask):
        # logits: (batch, seq_len, dim)
        # mention_select: (batch, mention_num, seq_len)
        # mention_mask: (batch, mention_num)
        # node_select: (batch, node_num, mention_num)
        # node_mask: (batch, node_num)
        _, _, dim = logits.shape
        batch_size, mention_num, seq_len = mention_select.shape
        _, node_num, _  = node_select.shape

        mention_select_rpt = mention_select.unsqueeze(3).repeat(1, 1, 1, dim)
        logit_rpt = logits.unsqueeze(1).repeat(1, mention_num, 1, 1)
        mention_reprs = torch.sum(logit_rpt * mention_select_rpt, 2) # (batch, mention_num, dim)

        avg_node_select = node_select / (torch.sum(node_select, 2) + 0.001).unsqueeze(2)
        node_select_rpt = avg_node_select.unsqueeze(3).repeat(1, 1, 1, dim)
        node_reprs = torch.sum(mention_reprs * node_select_rpt, 2) # batch, node_num, dim

        return node_reprs
    
    def compute_alignment_loss(self, node_reprs, text_reprs, node_mask):
        node_reprs_proj = self.graph_linear(node_reprs)
        text_reprs_proj = self.token_linear(text_reprs)
        cos_loss = - torch.sum(self.cosine(node_reprs_proj, text_reprs_proj) * node_mask) / (torch.sum(node_mask) + 1)
        return cos_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        ie_labels: Optional[torch.LongTensor] = None,
        mention_select: Optional[torch.FloatTensor] = None,
        mention_mask: Optional[torch.FloatTensor] = None,
        node_select: Optional[torch.FloatTensor] = None,
        node_mask: Optional[torch.FloatTensor] = None,
        graphs = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], LEDSeq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        encoder_outputs = self.led.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
        # computing ie loss
        batch_size, seq_len, dim = encoder_outputs.shape
        ie_logits = self.ie_linear(encoder_outputs).view(batch_size * seq_len, -1)
        ie_loss = nn.CrossEntropyLoss()(ie_logits, ie_labels.view(batch_size * seq_len))
        if torch.isnan(ie_loss):
            ie_loss = torch.new_zeros(1)[0]
        
        node_embeddings = self.get_node_embeddings(encoder_outputs, mention_select, mention_mask, node_select, node_mask)
        batch_size, max_node_num, dim = node_embeddings.shape
        node_nums = torch.sum(node_mask, 1).long().tolist()

        # assign the node embeddings into each dgl graph.
        for i in range(len(graphs)):
            graph_i = graphs[i]
            graph_i.ndata["hn"] = node_embeddings[i][0: node_nums[i]]
        
        batch_graph = dgl.batch(graphs)
        processed_batch_graph = self.gat(batch_graph)
        unbatched_graphs = dgl.unbatch(processed_batch_graph)

        updated_reprs = node_embeddings.new_zeros(batch_size, max_node_num, dim)
        for i in range(batch_size):
            updated_reprs[i][0: node_nums[i]] = unbatched_graphs[i].ndata["h"]
        graph_reprs = updated_reprs.mean(1).unsqueeze(1)

        align_loss = self.compute_alignment_loss(updated_reprs, node_embeddings, node_mask)

        outputs = self.led.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs+graph_reprs,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return {
            "ie_loss": ie_loss,
            "align_loss": align_loss,
            "lm_logits": lm_logits
        }


class PRIMERSummarizer(pl.LightningModule):
    def __init__(self, args):
        super(PRIMERSummarizer, self).__init__()
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.primer_path)
        self.model = GraphLED.from_pretrained(args.primer_path)
        self.model.gradient_checkpointing_enable()

        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_ddp = args.accelerator == "ddp"
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")

    def forward(self, input_ids, output_ids, labels, mention_select, mention_mask, node_select, node_mask, graphs):
        decoder_input_ids = output_ids[:, :-1]
        global_attention_mask = torch.zeros_like(input_ids).cuda()
        global_attention_mask[:, 0] = 1
        if self.args.join_method == "concat_start_wdoc_global":
            global_attention_mask[input_ids == self.docsep_token_id] = 1

        outputs = self.model(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            ie_labels=labels,
            mention_select=mention_select,
            mention_mask=mention_mask,
            node_select=node_select,
            node_mask=node_mask,
            graphs=graphs,
            global_attention_mask=global_attention_mask,
            use_cache=False,
        )
        lm_logits, ie_loss, align_loss = outputs["lm_logits"], outputs["ie_loss"], outputs["align_loss"]
        return lm_logits, ie_loss, align_loss

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
    def shared_step(self, batch):
        input_ids, output_ids, ie_labels, mention_select, mention_mask, node_select, node_mask, graphs = batch["input_ids"], batch["output_ids"], batch["labels"], batch["mention_select"], batch["mention_mask"], batch["node_select"], batch["node_mask"], batch["graphs"]

        if output_ids.shape[1] > self.args.max_length_tgt:
            new_output_ids = output_ids[:, 0:self.args.max_length_tgt]
            new_output_ids[:, -1] = 2
        else:
            new_output_ids = output_ids

        lm_logits, ie_loss, align_loss = self.forward(input_ids, new_output_ids, ie_labels, mention_select, mention_mask, node_select, node_mask, graphs)
        labels = output_ids[:, 1:].clone()

        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )

        total_loss = self.args.summ_weight * loss + self.args.ie_weight * ie_loss + self.args.align_weight * align_loss

        return total_loss

    def training_step(self, batch, batch_idx):
        input_ids, output_ids = batch["input_ids"], batch["output_ids"]
        loss = self.shared_step(batch)

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size": input_ids.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        return loss

    def compute_rouge_batch(self, input_ids, output_ids, gold_str):
        # all docs: list of length: per_device_batch_size
        dumped_results = []
        scorer = load_metric("rouge")
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            length_penalty=self.args.length_penalty,
            no_repeat_ngram_size=3 if self.args.applyTriblck else None,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        idxs_list = idxs.tolist()
        for i in range(len(gold_str)):
            json_result = {
                "id": idxs_list[i],
                "documents": all_docs[i],
                "generated_summary": generated_str[i],
                "gold_summary": gold_str[i]
            }
            dumped_results.append(deepcopy(json_result))

        result_batch = []
        for ref, pred in zip(gold_str, generated_str):
            s = scorer.compute(
                predictions=[pred],
                references=[ref],
                use_stemmer=True,
            )
            result_batch.append(
                (
                    s["rouge1"][0].recall,
                    s["rouge1"][0].precision,
                    s["rouge1"][0].fmeasure,
                    s["rouge2"][0].recall,
                    s["rouge2"][0].precision,
                    s["rouge2"][0].fmeasure,
                    s["rougeL"][0].recall,
                    s["rougeL"][0].precision,
                    s["rougeL"][0].fmeasure,
                    s["rougeLsum"][0].recall,
                    s["rougeLsum"][0].precision,
                    s["rougeLsum"][0].fmeasure,
                )
            )

        return result_batch, dumped_results

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False

        loss = self.shared_step(batch)
        return {"vloss": loss}

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        overall_dumps = []
        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss)
        metrics = [vloss]
        names = ["vloss"]

        logs = dict(zip(*[names, metrics]))
        self.logger.log_metrics(logs, step=self.global_step)
        return {
            "avg_val_loss": vloss,
            "log": logs,
            "progress_bar": logs,
        }

    def test_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        idxs, input_ids, output_ids, all_docs, tgt = batch["idxs"], batch["input_ids"], batch["output_ids"], batch["all_docs"], batch["tgt"]
        idxs_list = idxs.tolist()

        global_attention_mask = torch.zeros_like(input_ids).cuda()
        global_attention_mask[:, 0] = 1
        if self.args.join_method == "concat_start_wdoc_global":
            global_attention_mask[input_ids == self.docsep_token_id] = 1

        dumped_results = []

        generated_ids = self.model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            length_penalty=self.args.length_penalty,
            no_repeat_ngram_size=3 if self.args.applyTriblck else None,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        for i in range(len(tgt)):
            json_result = {
                "id": idxs_list[i],
                "documents": all_docs[i],
                "generated_summary": generated_str[i],
                "gold_summary": tgt[i]
            }
            dumped_results.append(deepcopy(json_result))

        return {"dumps": dumped_results}

    def test_epoch_end(self, outputs):
        overall_dumps = []
        for output in outputs:
            for dump in output["dumps"]:
                overall_dumps.append(dump)

        test_json_dir = os.path.join(self.args.model_path, "test_dumps")
        if not os.path.exists(test_json_dir):
            os.mkdir(test_json_dir)
        
        for dump_json in overall_dumps:
            with open(os.path.join(test_json_dir, str(dump_json["id"])+".json"), 'w', encoding="utf-8") as f:
                f.write(json.dumps(dump_json, indent=4))


def train(args):
    args.compute_rouge = True

    if args.resume_ckpt:
        model = PRIMERSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = PRIMERSummarizer(args)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.model_path, "summ_checkpoints")
    
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{avgr:.4f}",
        save_top_k=args.saveTopK,
        monitor="vloss",
        mode="min",
    )

    # initialize logger
    logger = TensorBoardLogger(os.path.join(args.model_path, "tb_logs"), name="my_model")

    # load datasets
    train_dataloader = get_dataloader_summ(
        args, args.dataset_name, model.tokenizer, "train", args.num_workers, True
    )
    valid_dataloader = get_dataloader_summ(
        args, args.dataset_name, model.tokenizer, "validation", args.num_workers, True
    )

    progress_bar = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate * args.acc_batch)
    # initialize trainer
    trainer = pl.Trainer(
        track_grad_norm=-1,
        max_steps=args.total_steps,
        replace_sampler_ddp=True,
        accumulate_grad_batches=args.acc_batch,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=1
        if args.num_train_data > 100 or args.num_train_data == -1
        else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, progress_bar],
        enable_checkpointing=True,
        precision=32 if args.fp32 else 16,
        strategy=DDPStrategy(find_unused_parameters=False) if args.accelerator == "ddp" else None,
        accelerator="gpu",
        devices=args.gpus
    )
    
    trainer.fit(model, train_dataloader, valid_dataloader)

    with open(os.path.join(args.model_path, "best_model_path.json"), 'w', encoding="utf-8") as f2:
        f2.write(json.dumps({"best_model": checkpoint_callback.best_model_path}))


def test(args):
    if args.resume_ckpt:
        model = PRIMERSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    elif args.name != "default":
        with open(os.path.join(args.model_path, "best_model_path.json"), 'r', encoding="utf-8") as f:
            best_model_path = json.loads(f.read())["best_model"]
        model = PRIMERSummarizer.load_from_checkpoint(best_model_path, args=args)
    else:
        model = PRIMERSummarizer(args)

    trainer = pl.Trainer(
        track_grad_norm=-1,
        max_steps=args.total_steps,
        replace_sampler_ddp=True,
        accumulate_grad_batches=args.acc_batch,
        log_every_n_steps=5,
        precision=32 if args.fp32 else 16,
        strategy=DDPStrategy(find_unused_parameters=False) if args.accelerator == "ddp" else None,
        accelerator="gpu",
        devices=args.gpus
    )

    # load dataset
    test_dataloader = get_dataloader_summ(
        args, args.dataset_name, model.tokenizer, "test", args.num_workers, False
    )
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Frequently tuned parameters
    parser.add_argument("--gpus", default=0, type=int, help="number of gpus to use")
    parser.add_argument("--primer_path", type=str, default="allenai/PRIMERA")
    parser.add_argument("--tokenizer", type=str, default="allenai/PRIMERA")
    parser.add_argument("--accelerator", default="ddp", type=str, help="Type of accelerator")
    parser.add_argument("--mode", default="train", choices=["pretrain", "train", "test"])
    parser.add_argument("--label_smoothing", type=float, default=0.1, required=False)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers to use for dataloader")
    parser.add_argument("--num_train_data", type=int, default=-1, help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--num_valid_data", type=int, default=-1, help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--num_test_data", type=int, default=-1, help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=256, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float, help="Number of steps to evaluate")
    parser.add_argument("--dataset_name", type=str, default="multi_news")
    parser.add_argument("--total_steps", type=int, default=25000, help="Number of steps to train")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Number of warmup steps")
    parser.add_argument("--compute_rouge", action="store_true", help="whether to compute rouge in validation steps")
    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--acc_batch", type=int, default=2, help="accumulated batch.")
    parser.add_argument("--setting", type=str, default="baseline", help="baseline, multitask, single_graph, multi_graph")

    # Other parameters
    parser.add_argument("--saveRouge", action="store_true", help="whether to compute rouge in validation steps")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--saveTopK", default=15, type=int)
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", default=None)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--join_method", type=str, default="concat_start_wdoc_global")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--attention_mode", type=str, default="sliding_chunks", help="Longformer attention mode")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
    parser.add_argument("--adafactor", action="store_true", help="Use adafactor optimizer")
    parser.add_argument("--fp32", action="store_true", help="default is fp16. Use --fp32 to switch to fp32")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--rand_seed", type=int, default=0, help="seed for random sampling, useful for few shot learning")
    parser.add_argument("--limit_valid_batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--report_steps", type=int, default=50, help="Number of report steps")
    parser.add_argument("--accum_data_per_step", type=int, default=16, help="Number of data per step")
    parser.add_argument("--remove_masks", action="store_true", help="remove all the masks in pretraining")
    parser.add_argument("--fix_lr", action="store_true", help="use fix learning rate")
    parser.add_argument("--test_immediate", action="store_true", help="test on the best checkpoint")
    parser.add_argument("--fewshot", action="store_true", help="whether this is a run for few shot learning")
    parser.add_argument("--eval_steps", type=int, default=2500, help="Number of steps to evaluate in the pre-training stage.")
    parser.add_argument("--valid_dumps_dir", type=str, default="default")
    parser.add_argument("--limit_test_batches", type=int, default=None, help="Number of batches to test in the test mode.")
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="length penalty, <1 encourage shorter message and >1 encourage longer messages")
    parser.add_argument("--mask_num", type=int, default=0, help="Number of masks in the input of summarization data")
    parser.add_argument("--test_batch_size", type=int, default=-1, help="batch size for test, used in few shot evaluation.")
    parser.add_argument("--applyTriblck", action="store_true", help="whether apply trigram block in the evaluation phase")
    parser.add_argument("--model_path", type=str, default="./checkpoints/")

    parser.add_argument("--summ_weight", type=float, default=1.0)
    parser.add_argument("--align_weight", type=float, default=0.1)
    parser.add_argument("--ie_weight", type=float, default=0.1)


    args = parser.parse_args()  # Get pad token id

    print("running experiments on " + args.dataset_name + " " + args.setting)
    args.data_path = os.path.join(args.data_path)
    args.model_path = os.path.join(args.model_path, args.name)
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with open(os.path.join(args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    else:
        test(args)
