# coding=utf-8

from __future__ import absolute_import, division, print_function
import random
import modeling
from modeling import PASentConfig
import argparse
import csv
import logging
import os
import random
import sys

import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from scipy.stats import spearmanr, pearsonr
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
import data_process.data_processor as data_processor

glue_processors = {
    "cola": data_processor.ColaProcessor,
    "mnli": data_processor.MnliProcessor,
    "mnli-mm": data_processor.MnliMismatchedProcessor,
    "mrpc": data_processor.MrpcProcessor,
    "sst-2": data_processor.Sst2Processor,
    "sts-b": data_processor.StsbProcessor,
    "qqp": data_processor.QqpProcessor,
    "qnli": data_processor.QnliProcessor,
    "rte": data_processor.RteProcessor,
    "wnli": data_processor.WnliProcessor,
    "snli": data_processor.SnliProcessor
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import sklearn.metrics as mtc


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def do_random_chunk(chunks_len):
    chunk_number = 0
    for (index, id) in enumerate(chunks_len):
        if id == 0:
            chunk_number = index
            break
    tmpt_chunks_len = chunks_len[0:chunk_number]
    random.shuffle(tmpt_chunks_len)
    shuffled_chunks_len = tmpt_chunks_len + [0] * (len(chunks_len) - chunk_number)
    return shuffled_chunks_len


def do_equal_chunk(chunks_len, max_len=3):
    total_chunk_len = 0
    for (index, id) in enumerate(chunks_len):
        total_chunk_len += id
    chunk_number = total_chunk_len // max_len + 1
    tmpt_chunks_len = [max_len] * chunk_number
    tmpt_chunks_len[-1] = total_chunk_len - max_len * (total_chunk_len // max_len)
    equal_chunks_len = tmpt_chunks_len + [0] * (len(chunks_len) - chunk_number)
    return equal_chunks_len


def Fscore(out, labels):
    outputs = np.argmax(out, axis=1)
    TP = np.sum(np.multiply(labels, outputs))
    FP = np.sum(np.logical_and(np.equal(labels, 0), np.equal(outputs, 1)))
    FN = np.sum(np.logical_and(np.equal(labels, 1), np.equal(outputs, 0)))
    TN = np.sum(np.logical_and(np.equal(labels, 0), np.equal(outputs, 0)))
    return TP, FP, FN, TN


def mcc(out, labels):
    return mtc.matthews_corrcoef(out, labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="data_process/data/glue_data",
                        type=str,
                        help="The input data features dir.")
    parser.add_argument("--dict_dir",
                        default="",
                        type=str,
                        help="the n-gram dict to parse the sentence")
    parser.add_argument("--bert_model_path", default='cached_model/albert-xxlarge-v2', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased-local, "
                             "bert-large, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--chunk_encoder", default='multi_cnn', type=str,
                        help="chunk encoder model")
    parser.add_argument("--do_concat",
                        action='store_true',
                        help="Whether to concat bert output.")
    parser.add_argument("--do_baseline",
                        action='store_true',
                        help="Whether to run baseline.")
    parser.add_argument("--do_store",
                        action='store_true',
                        help="Whether to store model.")
    parser.add_argument("--task_name",
                        default="snli",
                        type=str,
                        help="Whether to concat bert output.")
    parser.add_argument("--bert_type",
                        default="albert",
                        type=str,
                        help="The type of transformer we want to run.")
    parser.add_argument("--hidden_size",
                        default=1024,
                        type=int,
                        help="hidden size of the model")
    parser.add_argument("--max_chunk_number",
                        default=64,
                        type=int,
                        help="Max chunk number for the sentence.")
    parser.add_argument("--max_chunk_len",
                        default=16,
                        type=int,
                        help="Max word piece number in the chunk")
    parser.add_argument("--output_dir",
                        default="multi_cnn_albert_large",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_random",
                        action='store_true',
                        help="Whether to make random chunk.")
    parser.add_argument("--do_dev",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".pytorch_pretrained_bert",
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--mode', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task_name = args.task_name.lower()
    if task_name not in glue_processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = glue_processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    num_train_optimization_steps = None
    if args.do_train:
        if args.do_dev:
            train_features = processor.get_dev_features_from_examples(data_dir=args.data_dir, bert_type=args.bert_type,
                                                                      max_chunk_number=args.max_chunk_number,
                                                                      ngram_path=args.dict_dir,max_seq_len=args.max_seq_length)
        else:
            train_features = processor.get_train_features_from_examples(data_dir=args.data_dir,
                                                                        bert_type=args.bert_type,
                                                                        max_chunk_number=args.max_chunk_number,
                                                                        ngram_path=args.dict_dir,max_seq_len=args.max_seq_length)
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    config = PASentConfig(bert_model_path=args.bert_model_path, bert_model_type=args.bert_type,
                          chunk_encoder=args.chunk_encoder,
                          hidden_size=args.hidden_size, do_baseline=args.do_baseline,
                          do_concat=args.do_concat, max_chunk_number=args.max_chunk_number,
                          max_chunk_len=args.max_chunk_len, label_number=num_labels, device=device)
    model = modeling.PASentForSequenceClassification(config)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            import apex.amp as amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    best_epoch = 0
    best_result = 0.0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        for f in train_features:
            if len(f.input_ids) != args.max_seq_length:
                print(len(f.input_ids))

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
        if args.do_random:
            all_chunk_lens = torch.tensor([do_random_chunk(f.chunks_len) for f in train_features], dtype=torch.long)
        else:
            all_chunk_lens = torch.tensor([f.chunks_len for f in train_features], dtype=torch.long)
        if args.task_name == "sts-b":
            all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.float)
        else:
            all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                   all_chunk_lens, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        eval_dataloader = None
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            eval_features = processor.get_eval_features_from_examples(data_dir=args.data_dir, bert_type=args.bert_type,
                                                                      max_chunk_number=args.max_chunk_number,
                                                                      ngram_path=args.dict_dir,max_seq_len=args.max_seq_length)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
            all_chunk_lens = torch.tensor([f.chunks_len for f in eval_features], dtype=torch.long)

            if args.task_name == "sts-b":
                all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.float)
            else:
                all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                      all_chunk_lens, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, chunk_lens, \
                label_ids = batch
                loss = model(input_ids, attention_mask, token_type_ids, chunk_lens, label_ids)[0]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                ##print(loss.item())
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, str(epoch) + "_pytorch_model.bin")
            if args.do_store:
                torch.save(model_to_save.state_dict(), output_model_file)
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                model.eval()
                model.to(device)
                logger.info("***** Running dev evaluation *****")
                logger.info("  Num examples = %d", len(eval_features))
                logger.info("  Batch size = %d", args.eval_batch_size)
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                total_pred, total_labels = [], []
                total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
                # print('# model parameters:', sum(param.numel() for param in predict_model.parameters()))
                output_logits_file = os.path.join(args.output_dir, str(epoch) + "_eval_logits_results.tsv")
                with open(output_logits_file, "w") as writer:
                    writer.write(
                        "index" + "\t" + "\t".join(["logits " + str(i) for i in range(len(label_list))]) + "\n")
                    for batch in tqdm(eval_dataloader, desc="Evaluating"):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, attention_mask, token_type_ids, chunk_lens, \
                        label_ids = batch
                        with torch.no_grad():
                            tmp_eval_loss, logits = model(input_ids, attention_mask, token_type_ids, chunk_lens,
                                                          label_ids)
                        label_ids = label_ids.squeeze()
                        logits = logits.squeeze()

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1
                        if task_name == "sts-b":
                            for i, logit in enumerate(logits):
                                if logit < 0:
                                    logits[i] = np.float(0)
                                if logit > 5:
                                    logits[i] = np.float(5)
                            total_pred.extend(logits.squeeze().tolist())
                            total_labels.extend(label_ids.squeeze().tolist())
                        else:
                            tmp_eval_accuracy = accuracy(logits, label_ids)
                            eval_loss += tmp_eval_loss.mean().item()
                            eval_accuracy += tmp_eval_accuracy
                        index = 0

                        #print(logits)
                        if task_name == "cola":
                            total_pred.extend(np.argmax(logits, axis=1).squeeze().tolist())
                            total_labels.extend(label_ids.squeeze().tolist())
                        elif task_name == "mrpc" or task_name == "qqp":  # need F1 score
                            TP, FP, FN, TN = Fscore(logits, label_ids)
                            total_TP += TP
                            total_FP += FP
                            total_FN += FN
                            total_TN += TN
                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                loss = tr_loss / nb_tr_steps if args.do_train else None
                result = {'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          'global_step': global_step,
                          'loss': loss}

                if task_name == "cola":
                    eval_mcc = mcc(total_pred, total_labels)
                    result["eval_mcc"] = eval_mcc
                elif task_name == "mrpc" or task_name == "qqp":  # need F1 score
                    if total_TP + total_FP == 0:
                        P = 0
                    else:
                        P = total_TP / (total_TP + total_FP)
                    if total_TP + total_FN == 0:
                        R = 0
                    else:
                        R = total_TP / (total_TP + total_FN)
                    if P + R == 0:
                        F1 = 0
                    else:
                        F1 = 2.00 * P * R / (P + R)
                    result["Precision"] = P
                    result["Recall"] = R
                    result["F1 score"] = F1
                elif task_name == "sts-b":  # need F1 score
                    result["pearson corref"] = pearsonr(total_pred, total_labels)
                    result["spearman corref"] = spearmanr(total_pred, total_labels)
                print(result)
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
                        writer.write("Epoch: %s, %s = %s\n" % (str(epoch), key, str(result[key])))
                if task_name == 'snli':
                    model.eval()
                    model.to(device)
                    processor = glue_processors['snli']()
                    test_features = processor.get_test_features_from_examples(data_dir=args.data_dir,
                                                                             bert_type=args.bert_type,
                                                                             max_chunk_number=args.max_chunk_number,
                                                                             ngram_path=args.dict_dir,
                                                                             max_seq_len=args.max_seq_length)
                    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                    all_attention_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
                    all_token_type_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
                    all_chunk_lens = torch.tensor([f.chunks_len for f in test_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label for f in test_features], dtype=torch.long)
                    test_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                              all_chunk_lens, all_label_ids)
                    # Run prediction for full data
                    test_sampler = SequentialSampler(test_data)
                    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
                    logger.info("***** Running snli_test evaluation *****")
                    logger.info("  Num examples = %d", len(eval_features))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    output_logits_file = os.path.join(args.output_dir, str(epoch) + "snli_test_logits_results.tsv")
                    with open(output_logits_file, "w") as writer:
                        writer.write(
                            "index" + "\t" + "\t".join(["logits " + str(i) for i in range(len(label_list))]) + "\n")
                        for batch in tqdm(test_dataloader, desc="Evaluating"):
                            batch = tuple(t.to(device) for t in batch)
                            input_ids, attention_mask, token_type_ids, chunk_lens, \
                            label_ids = batch
                            with torch.no_grad():
                                tmp_eval_loss, logits = model(input_ids, attention_mask, token_type_ids, chunk_lens,
                                                              label_ids)
                            label_ids = label_ids.squeeze()
                            logits = logits.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            tmp_eval_accuracy = accuracy(logits, label_ids)
                            eval_loss += tmp_eval_loss.mean().item()
                            eval_accuracy += tmp_eval_accuracy
                            index = 0
                            for (i, prediction) in enumerate(logits):
                                prediction = prediction.tolist()
                                # print(prediction)
                                writer.write(str(index) + "\t" + "\t".join([str(pred) for pred in prediction]) + "\n")
                                index += 1
                            nb_eval_examples += input_ids.size(0)
                            nb_eval_steps += 1
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'loss': loss}
                    print(result)
                    output_eval_file = os.path.join(args.output_dir, "snli_test_eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
                            writer.write("Epoch: %s, %s = %s\n" % (str(epoch), key, str(result[key])))
                if task_name == 'mnli':
                    model.eval()
                    model.to(device)
                    processor = glue_processors['mnli-mm']()
                    test_features = processor.get_dev_features_from_examples(data_dir=args.data_dir, bert_type=args.bert_type,
                                                                      max_chunk_number=args.max_chunk_number,
                                                                      ngram_path=args.dict_dir,max_seq_len=args.max_seq_length)
                    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                    all_attention_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
                    all_token_type_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
                    all_chunk_lens = torch.tensor([f.chunks_len for f in test_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label for f in test_features], dtype=torch.long)
                    test_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                              all_chunk_lens, all_label_ids)
                    # Run prediction for full data
                    test_sampler = SequentialSampler(test_data)
                    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
                    logger.info("***** Running mmdev evaluation *****")
                    logger.info("  Num examples = %d", len(eval_features))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    output_logits_file = os.path.join(args.output_dir, str(epoch) + "mm_eval_logits_results.tsv")
                    with open(output_logits_file, "w") as writer:
                        writer.write(
                            "index" + "\t" + "\t".join(["logits " + str(i) for i in range(len(label_list))]) + "\n")
                        for batch in tqdm(test_dataloader, desc="Evaluating"):
                            batch = tuple(t.to(device) for t in batch)
                            input_ids, attention_mask, token_type_ids, chunk_lens, \
                            label_ids = batch
                            with torch.no_grad():
                                tmp_eval_loss, logits = model(input_ids, attention_mask, token_type_ids, chunk_lens,
                                                              label_ids)
                            label_ids = label_ids.squeeze()
                            logits = logits.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            tmp_eval_accuracy = accuracy(logits, label_ids)
                            eval_loss += tmp_eval_loss.mean().item()
                            eval_accuracy += tmp_eval_accuracy
                            index = 0
                            for (i, prediction) in enumerate(logits):
                                prediction = prediction.tolist()
                                # print(prediction)
                                writer.write(str(index) + "\t" + "\t".join([str(pred) for pred in prediction]) + "\n")
                                index += 1
                            nb_eval_examples += input_ids.size(0)
                            nb_eval_steps += 1
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples
                    loss = tr_loss / nb_tr_steps if args.do_train else None
                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'loss': loss}
                    print(result)
                    output_eval_file = os.path.join(args.output_dir, "mm_eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))
                            writer.write("Epoch: %s, %s = %s\n" % (str(epoch), key, str(result[key])))

            if args.do_predict:
                model.eval()
                model.to(device)
                test_features =  processor.get_test_features_from_examples(data_dir=args.data_dir, bert_type=args.bert_type,
                                                                      max_chunk_number=args.max_chunk_number,
                                                                      ngram_path=args.dict_dir,max_seq_len=args.max_seq_length)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(test_features))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                all_attention_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
                all_token_type_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
                all_chunk_lens = torch.tensor([f.chunks_len for f in test_features], dtype=torch.long)
                test_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                          all_chunk_lens)
                # Run prediction for full data
                test_sampler = SequentialSampler(test_data)
                test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
                predictions = []
                output_logits_file = os.path.join(args.output_dir, str(epoch) + "_logits_results.tsv")
                with open(output_logits_file, "w") as writer:
                    for input_ids, attention_mask, token_type_ids, chunk_lens in tqdm(
                            test_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        token_type_ids = token_type_ids.to(device)
                        chunk_lens = chunk_lens.to(device)
                        with torch.no_grad():
                            logits = model(input_ids, attention_mask, token_type_ids, chunk_lens,
                                           None)
                        logits = logits.squeeze()
                        logits = logits.detach().cpu().numpy()
                        if task_name == 'sts-b':
                            for i, logit in enumerate(logits):
                                if logit < 0:
                                    logits[i] = np.float(0)
                                if logit > 5:
                                    logits[i] = np.float(5)
                            for (i, prediction) in enumerate(logits):
                                predictions.append(prediction)
                        else:
                            for (i, prediction) in enumerate(logits):
                                predict_label = np.argmax(prediction)
                                predictions.append(predict_label)
                output_predict_file = os.path.join(args.output_dir, str(epoch) + "_pred_results.tsv")
                index = 0
                with open(output_predict_file, "w") as writer:
                    writer.write("index" + "\t" + "prediction" + "\n")
                    for pred in predictions:
                        if task_name == "sts-b":
                            writer.write(str(index) + "\t" + ('%.4f' % pred) + "\n")
                        else:
                            writer.write(str(index) + "\t" + str(label_list[int(pred)]) + "\n")
                        index += 1

                if task_name == 'mnli':
                    model.eval()
                    model.to(device)
                    processor = glue_processors['mnli-mm']()
                    test_features =  processor.get_test_features_from_examples(data_dir=args.data_dir, bert_type=args.bert_type,
                                                                      max_chunk_number=args.max_chunk_number,
                                                                      ngram_path=args.dict_dir,max_seq_len=args.max_seq_length)
                    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                    all_attention_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
                    all_token_type_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
                    all_chunk_lens = torch.tensor([f.chunks_len for f in test_features], dtype=torch.long)
                    test_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                              all_chunk_lens)
                    # Run prediction for full data
                    test_sampler = SequentialSampler(test_data)
                    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
                    logger.info("***** Running mmdev evaluation *****")
                    logger.info("  Num examples = %d", len(test_dataloader))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    output_logits_file = os.path.join(args.output_dir, str(epoch) + "mm_eval_logits_results.tsv")
                    with open(output_logits_file, "w") as writer:
                        writer.write(
                            "index" + "\t" + "\t".join(["logits " + str(i) for i in range(len(label_list))]) + "\n")
                        for batch in tqdm(test_dataloader, desc="Evaluating"):
                            batch = tuple(t.to(device) for t in batch)
                            input_ids, attention_mask, token_type_ids, chunk_lens = batch
                            with torch.no_grad():
                                logits = model(input_ids, attention_mask, token_type_ids, chunk_lens, None)
                            logits = logits.detach().cpu().numpy()
                            index = 0
                            for (i, prediction) in enumerate(logits):
                                prediction = prediction.tolist()
                                # print(prediction)
                                writer.write(str(index) + "\t" + "\t".join([str(pred) for pred in prediction]) + "\n")
                                index += 1
                            nb_eval_examples += input_ids.size(0)
                            nb_eval_steps += 1
                    print(result)
                    output_predict_file = os.path.join(args.output_dir, str(epoch) + "mm_pred_results.tsv")
                    index = 0
                    with open(output_predict_file, "w") as writer:
                        writer.write("index" + "\t" + "prediction" + "\n")
                        for pred in predictions:
                            writer.write(str(index) + "\t" + str(label_list[int(pred)]) + "\n")
                            index += 1


if __name__ == "__main__":
    main()
