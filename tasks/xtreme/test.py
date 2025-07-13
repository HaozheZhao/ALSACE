#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reproduce the main evaluation in `Multitask Prompted Training Enables Zero-Shot Task Generalization` using PyTorch.

This script is heavily adapted from https://github.com/huggingface/transformers/blob/7533d30acd975027e83a548e4c38e06fa335291b/examples/pytorch/multiple-choice/run_swag_no_trainer.py
"""

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import json
import codecs
import datetime

import datasets
import torch
from datasets import load_dataset, load_metric, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter
import numpy as np

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
)
from transformers.file_utils import PaddingStrategy
from transformers import Adafactor

# from promptsource.templates_test import DatasetTemplates
from templates import DatasetTemplates
from torch.nn import CrossEntropyLoss
from torch import nn
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


TOP_K_for_each_task = {
    'super_glue/wsc.fixed': 10,
    'winogrande/winogrande_xl': 5,
    'anli/r1': 15,
    'anli/r2': 15,
    'anli/r3': 15,
    'super_glue/cb': 15,
    'super_glue/rte': 10,
    'hellaswag': 4,
    'super_glue/copa': 12,
    'super_glue/wic': 10,
    'story_cloze/2016': 5
}


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def read_split_list(file_name):
    test_task_list = []
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            task_tuple = line.split('/')
            if len(task_tuple) == 2:
                test_task_list.append(task_tuple)
            else:
                test_task_list.append((task_tuple[0], None))

    return test_task_list


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                              " sequences shorter will be padded if `--pad_to_max_lengh` is passed."), )
    parser.add_argument("--target_max_length", type=int, default=256,
                        help="Target max length. Sequences longer than this will be truncated." )
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.", )
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
                        required=True, )
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument("--template_dir", type=str, default='/home/yanan/shaonan/t-zero/templates_test', help="模版文件的位置", )
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).", )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode and run training only with a subset of data.", )
    parser.add_argument("--ga_dev_distribution", type=str, choices=['uniform', 'ratio'], default='uniform', help="ga_dev的分布", )
    parser.add_argument("--parallelize", action="store_true",
                        help=(
                            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
                            "Note that this feature is still experimental in HF Transformers."),
                        )
    parser.add_argument("--test_split", type=str, help='测试任务名单')
    parser.add_argument("--dataset_type", type=str, choices=['ga', 'pt', 'all'])

    # prompt tuning 的参数
    parser.add_argument("--lr", type=float, default=5e-5)   # PT 原文使用0.3
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help='单卡的batch size')
    parser.add_argument("--num_training_steps", type=int, default=1000)
    parser.add_argument("--eval_period", type=int, default=50, help='训练时eval的时间间隔(用来挑ckpt)')

    parser.add_argument("--only_train_single_template", action="store_true")
    parser.add_argument("--only_eval_single_template", action="store_true")

    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    # 这里得到的应该是一个batch的feature
    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])  # input_ids： exam_num * num_choice * seq_len
        # 输入之前是一个样本的多个选项一个dict，flatten后是[ [{exam1_opt1}, {exam1_opt2}], []  ]
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        # 把所有样本拍成一个list
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])   # 当前batch所有label的最大长度

        # padding 后的label
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


@dataclass
class DataCollatorForMultipleChoiceTraining:
    """
    用于训练数据
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    # 这里得到的应该是一个batch的feature
    def __call__(self, features):
        # num_choices = len(features[0]["input_ids"])  # input_ids： exam_num * num_choice * seq_len
        # 输入之前是一个样本的多个选项一个dict，flatten后是[ [{exam1_opt1}, {exam1_opt2}], []  ]
        # flattened_features = [
        #     [
        #         {
        #             k: v[i]
        #             for k, v in feature.items()
        #             if k != "targets"
        #         }
        #         for i in range(num_choices)
        #     ]
        #     for feature in features
        # ]
        # print(f'features: {features}')
        flattened_features = [
            {
                k: v
                for k, v in feature.items()
                if k != "targets"
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])  # 当前batch所有label的最大长度

        # padding 后的label
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        # 屏蔽pad位
        for idx, label_ids in enumerate(batch["labels"]):
            label_ids = [ids if ids != 0 else -100 for ids in label_ids]
            batch["labels"][idx] = label_ids

        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


# TODO
def build_pseudo_dataset(dataset_name, dataset_config_name, filtered_dataset, MAXNUM, distribution='uniform'):
    '''每次传进来之前，套上新的confidence和pseudo label, 只需要传一个数量进来，数量在主函数里算，这里返回的是raw dataset, 然后把下面那个做成prompt化程序即可'''
    task_name = f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}'
    dataset_distribution = {'anli/r1': {0: 334, 2: 333, 1: 333},
                            'anli/r2': {0: 334, 1: 333, 2: 333},
                            'anli/r3': {0: 402, 1: 402, 2: 396},
                            'super_glue/cb': {1: 28, 0: 23, 2: 5},
                            'super_glue/rte': {0: 146, 1: 131},
                            'super_glue/wsc.fixed': {0: 66, 1: 38},
                            'winogrande/winogrande_xl': {'2': 639, '1': 628},
                            'super_glue/copa': {0: 55, 1: 45},
                            'hellaswag': {'2': 2584, '0': 2515, '1': 2485, '3': 2458},
                            'super_glue/wic': {0: 319, 1: 319},
                            'story_cloze/2016': {1: 962, 2:909}
                            }

    label_key = 'label'
    if dataset_name in ['winogrande']:
        label_key = 'answer'
    if dataset_name in ['story_cloze']:
        label_key = 'answer_right_ending'

    label_list = filtered_dataset[label_key]
    label_type_set = set(label_list)
    print(f'label_type_set: {label_type_set}')
    ga_dataset_list = []
    dataset_distribution_of_pseudo_dataset = dict(Counter(filtered_dataset[label_key]))
    # 把每个类别的样本分类存储
    for label_type in label_type_set:
        single_label_dataset = filtered_dataset.filter(lambda x: x[label_key] == label_type)
        # we do not select randomly, instead, we choose the most confident part
        single_label_dataset = single_label_dataset.sort('confidence', reverse=True)
        single_label_dataset = single_label_dataset.remove_columns('confidence')
        
        # since we get the distribution of the pseudo label dataaset, we should use the new distribution instead of uniform
        if distribution == 'ratio':
            example_num_per_label = math.ceil(
                dataset_distribution_of_pseudo_dataset[label_type] / sum(dataset_distribution[task_name].values()) * MAXNUM)
        else:
            example_num_per_label = math.ceil(MAXNUM / len(label_type_set))
        selected_dataset = single_label_dataset.select(
            range(min(example_num_per_label, len(single_label_dataset))))
        selected_dataset = selected_dataset.shuffle(seed=42)
        ga_dataset_list.append(selected_dataset)

    # 每个class的样本train和eval各分一半
    # train_dataset_list = [ga_dataset.select(range(len(ga_dataset) // 2)) for ga_dataset in ga_dataset_list if len(ga_dataset) // 2 > 0]
    # dev_dataset_list = [ga_dataset.select(range(len(ga_dataset)//2, len(ga_dataset))) for ga_dataset in ga_dataset_list if len(ga_dataset) // 2 > 0]
    # combined_train_dataset = concatenate_datasets(train_dataset_list)
    # combined_dev_dataset = concatenate_datasets(dev_dataset_list)
    combined_ga_dataset = concatenate_datasets(ga_dataset_list)

    # print(f'combined_train_dataset: {combined_train_dataset}')
    # print(f'labels: {combined_train_dataset[label_key]}')
    # print(f'combined_dev_dataset: {combined_dev_dataset}')
    # print(f'labels: {combined_dev_dataset[label_key]}')
    print(f'combined_ga_dataset: {combined_ga_dataset}')
    print(f'labels: {combined_ga_dataset[label_key]}')
    # use train as dev
    # return combined_train_dataset, combined_dev_dataset
    return combined_ga_dataset


def build_pt_dataset(dataset_name, dataset_config_name, raw_datasets, distribution='uniform'):
    task_name = f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}'
    dataset_distribution = {'anli/r1': {0: 334, 2: 333, 1: 333},
                            'anli/r2': {0: 334, 1: 333, 2: 333},
                            'anli/r3': {0: 402, 1: 402, 2: 396},
                            'super_glue/cb': {1: 28, 0: 23, 2: 5},
                            'super_glue/rte': {0: 146, 1: 131},
                            'super_glue/wsc.fixed': {0: 66, 1: 38},
                            'winogrande/winogrande_xl': {'2': 639, '1': 628},
                            'super_glue/copa': {0: 55, 1: 45},
                            'hellaswag': {'2': 2584, '0': 2515, '1': 2485, '3': 2458},
                            'super_glue/wic': {0: 319, 1: 319},
                            'story_cloze/2016': {1: 962, 2:909}
                            }

    label_key = 'label'
    if dataset_name in ['winogrande']:
        label_key = 'answer'
    if dataset_name in ['story_cloze']:
        label_key = 'answer_right_ending'
    filtered_dataset = raw_datasets

    # 对anli数据集和winogrande的特别处理
    # if dataset_name == 'anli':
    #     print(f'len of raw_dataset: {filtered_dataset}')
    #     filtered_dataset = filtered_dataset.filter(lambda x: len(x['reason']) > 0)
    #     print(f'len of filtered_dataset: {filtered_dataset}')
    #     if dataset_config_name == 'r1':
    #         filtered_dataset = filtered_dataset.select(range(900))
    #     elif dataset_config_name == 'r2':
    #         filtered_dataset = filtered_dataset.select(range(1500))
    #     elif dataset_config_name == 'r3':
    #         index = list(range(len(filtered_dataset)))
    #         index = index[8000:]
    #         filtered_dataset = filtered_dataset.select(index)
    # if dataset_name == 'winogrande':
    #     filtered_dataset = load_dataset('winogrande', 'winogrande_debiased', split='train')

    label_list = filtered_dataset[label_key]
    label_type_set = set(label_list)
    print(f'label_type_set: {label_type_set}')
    ga_dataset_list = []
    MAXNUM = 32
    # 把每个类别的样本分类存储
    for label_type in label_type_set:
        single_label_dataset = filtered_dataset.filter(lambda x: x[label_key] == label_type)
        single_label_dataset = single_label_dataset.shuffle(seed=42)

        # if distribution == 'ratio':
        #     example_num_per_label = math.ceil(
        #         dataset_distribution[task_name][label_type] / sum(dataset_distribution[task_name].values()) * MAXNUM)
        # else:
        example_num_per_label = math.ceil(MAXNUM / len(label_type_set))

        ga_dataset_list.append(single_label_dataset.select(
            range(min(example_num_per_label, len(single_label_dataset)))))

    # 每个class的样本train和eval各分一半
    # train_dataset_list = [ga_dataset.select(range(len(ga_dataset) // 2)) for ga_dataset in ga_dataset_list]
    # dev_dataset_list = [ga_dataset.select(range(len(ga_dataset)//2, len(ga_dataset))) for ga_dataset in ga_dataset_list]

    # combined_train_dataset = concatenate_datasets(train_dataset_list)
    # combined_dev_dataset = concatenate_datasets(dev_dataset_list)
    combined_ga_dataset = concatenate_datasets(ga_dataset_list)

    # print(f'combined_train_dataset: {combined_train_dataset}')
    # print(f'labels: {combined_train_dataset[label_key]}')
    # print(f'combined_dev_dataset: {combined_dev_dataset}')
    # print(f'labels: {combined_dev_dataset[label_key]}')
    print(f'combined_ga_dataset: {combined_ga_dataset}')
    print(f'labels: {combined_ga_dataset[label_key]}')
    # return combined_train_dataset, combined_dev_dataset
    return combined_ga_dataset


def get_prompted_datasets(dataset_name, dataset_config_name, args, accelerator, preprocess_train_function, preprocess_eval_function, column_names, pseudo_label_datasets):
    '''input raw_dataset
    idx_subset: optional: if provided, get subset, else get 32
    return prompted dataset
    '''
    train_dataset_list = []
    prompts = DatasetTemplates(
                f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
                template_dir=args.template_dir)

    # key是uuid
    # template_list = prompts.name_to_id_mapping.keys()
    template_list = prompts.templates.keys()
    logger.info(f'{dataset_name}的模板列表：{template_list}')
    for template_id in template_list:
        
        template = prompts.templates[template_id]
        template_name = template.name

        logger.info(f'{template.metadata.original_task}, type: {type(template.metadata.original_task)}')
        if template.metadata.original_task is not True:
            logger.info(f'跳过{template_name}, 因为不是原始任务形式')
            continue

        # 过滤copa样本, 一些prompt只适用于部分样本
        filtered_pseudo_dataset = None
        if dataset_config_name == 'copa':
            if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
                filtered_pseudo_dataset = pseudo_label_datasets.filter(lambda example: example['question'] == 'effect')
            if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
                filtered_pseudo_dataset = pseudo_label_datasets.filter(lambda example: example['question'] == 'cause')
        # ga的32 dev在这里过滤，不然filter完可能少于32
        if not filtered_pseudo_dataset:
            filtered_pseudo_dataset = pseudo_label_datasets

        # 在这里返回一下idx
        # copa单独处理成先返回再得到subset的样子
        # 这样便于处理idx
        
        # ga的32 dev在这里过滤，不然filter完可能少于32
        
        # TODO
        
        print(f'使用数据 {dataset_name}_{dataset_config_name}_{template_name}')

        with accelerator.main_process_first():
            single_train_processed_dataset = filtered_pseudo_dataset.map(
                preprocess_train_function, fn_kwargs={"template": template}, batched=True, remove_columns=column_names,
                load_from_cache_file=False)
            
            train_dataset_list.append(single_train_processed_dataset)

    # 开始准备训练
    all_train_dataset = concatenate_datasets(train_dataset_list)
    return all_train_dataset


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state)

    test_task_list = read_split_list(args.test_split)
    logger.info(f'训练任务: {test_task_list}')

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        # T0 model
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    
    # 读数据集
    for dataset_name, dataset_config_name in test_task_list:
        dev_dataset_dict = {}
        
        # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if dataset_config_name:
            if dataset_name == 'story_cloze':
                data_path = os.path.join("/share/zongyu/zhoujing/huggingface_datasets", dataset_name, dataset_config_name, "validation")
            else:
                data_path = os.path.join("/share/zongyu/zhoujing/huggingface_datasets", dataset_name, dataset_config_name, "train")
        else:
            data_path = os.path.join("/share/zongyu/zhoujing/huggingface_datasets", dataset_name, "train")
        train_datasets = load_from_disk(data_path)
        if dataset_config_name:
            if dataset_name == 'story_cloze':
                data_path = os.path.join("/share/zongyu/zhoujing/huggingface_datasets", dataset_name, dataset_config_name, "test")
            else:
                data_path = os.path.join("/share/zongyu/zhoujing/huggingface_datasets", dataset_name, dataset_config_name, "validation")
        else:
            data_path = os.path.join("/share/zongyu/zhoujing/huggingface_datasets", dataset_name, "validation")
        dev_datasets = load_from_disk(data_path)
        
        # TODO: column names for train and dev
        # logger.info(f'raw dataset for {dataset_name}_{dataset_config_name}: {raw_datasets}')

        column_names = train_datasets.column_names
        logger.info(f'column name: {column_names}')
        # this is the old preprocess_fun wo soft-embedding
        def preprocess_function(examples, template):
            bs = len(examples[column_names[0]])
            input_texts = []
            target_texts = []
            answer_choices_texts = []
            idx = []
            for i in range(bs):
                ex = {k: examples[k][i] for k in column_names}
                outputs = template.apply(ex)
                if len(outputs) == 2:
                    input, target = outputs
                else:
                    assert (len(outputs) == 1 and len(outputs[0]) == 0)
                    continue
                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)
                idx.append(ex['id'])

            bs = len(input_texts)
            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=True,
            )

            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]
            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }
            
            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]
            features["id"] = idx
            return features

        def preprocess_eval_function(examples, template):
            bs = len(examples[column_names[0]])  # map的回调函数一次处理一个batch数据(跟训练的batch不一样)

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                # 样本的每个字段组合起来
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                # print(f'debug: ex: {ex}')
                input, target = template.apply(ex)

                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)  # 列表的每个元素是该样本的选项列表

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            # input只编码一次, 选项每个都要编码
            # exam * num_choice * label_len
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]

            # k 是input ids/attention_mask等
            # 这里的意思就是是把input_ids等复制选项个数的次数
            # input_ids: [ [exam1, exam1], [exam2, exam2], ... ]
            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

            # label的ids： exam_num * label_len
            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]

            # label的attention mask
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]

            # targets 是答案文本
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]
            return features

        def preprocess_train_function(examples, template):
            bs = len(examples[column_names[0]])

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                # 样本的每个字段组合起来
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                # print(f'debug: ex: {ex}')
                input, target = template.apply(ex)

                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            # input只编码一次, 选项每个都要编码
            tokenized_targets = tokenizer(
                    target_texts,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )

            # k 是input ids/attention_mask等
            # 这里的意思就是是把input_ids等复制选项个数的次数
            features = tokenized_inputs

            # label的ids
            features["labels"] = tokenized_targets["input_ids"]

            # label的attention mask
            features["labels_attention_mask"] = tokenized_targets["attention_mask"]

            # targets 是答案的id
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]
            return features

        # Get the prompt to apply and the possible targets.
        logger.info(f'use template_dir: {args.template_dir}')

        prompts = DatasetTemplates(
            f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
            template_dir=args.template_dir)

        # key是uuid
        # template_list = prompts.name_to_id_mapping.keys()
        template_list = prompts.templates.keys()
        logger.info(f'{dataset_name}的模板列表：{template_list}')

        checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        task_name = dataset_name + '_' + dataset_config_name if dataset_config_name is not None else dataset_name
        for epoch in range(3):
            if args.model_name_or_path:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )
            else:
                logger.info("Training new model from scratch")
                model = AutoModelForSeq2SeqLM.from_config(config)

            # Use the device given by the `accelerator` object.
            device = accelerator.device
            if args.parallelize:
                assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
                model.parallelize()
            else:
                model.to(device)
            if epoch == 0:
                confidence, prompt_vote_pred = get_pseudo_label(args, dataset_name, dataset_config_name, template_list, model, tokenizer, dev_datasets, preprocess_function, accelerator, epoch)
                if dataset_name in ['winogrande', 'story_cloze']:
                    prompt_vote_pred += 1
                prompt_vote_pred = prompt_vote_pred.astype('int')
                if dataset_name in ['hellaswag', 'winogrande']:
                    prompt_vote_pred = prompt_vote_pred.astype('str')
                label_key = 'label'
                if dataset_name in ['winogrande']:
                    label_key = 'answer'
                elif dataset_name in ['story_cloze']:
                    label_key = 'answer_right_ending'
                print(dev_datasets[label_key])
                pseudo_label_datasets = dev_datasets.remove_columns(label_key)
                pseudo_label_datasets = pseudo_label_datasets.add_column(label_key, prompt_vote_pred.tolist())
                pseudo_label_datasets = pseudo_label_datasets.add_column('confidence', confidence.tolist())
                print(pseudo_label_datasets[label_key])
                pseudo_label_datasets = pseudo_label_datasets.flatten_indices()
                pseudo_label_datasets = build_pseudo_dataset(dataset_name, dataset_config_name, pseudo_label_datasets, 32)
                all_train_dataset = get_prompted_datasets(dataset_name, dataset_config_name, args, accelerator, preprocess_train_function, preprocess_eval_function, column_names, pseudo_label_datasets)
            # idx_subset is a np.array
            # 每次train的时候输入一个idx subset, 输出扩张后的sebset, 和该subset对应的pseudo_label (np.array)
            # 输入全量的dev set用来供训完的模型打标（原始的dev set需要保持不动）
            confidence, prompt_vote_pred = do_train(args, model, tokenizer, all_train_dataset, dev_datasets, accelerator, dataset_name, dataset_config_name, epoch, template_list, preprocess_function)
            # 所有sample都被打过标
            if 32 * np.power(5, epoch) > len(dev_datasets):
                break
            # 在do_train里打标，打完之后把打标idx_subset用来构建新的pseudo_subset
            if dataset_name in ['winogrande', 'story_cloze']:
                prompt_vote_pred += 1
            prompt_vote_pred = prompt_vote_pred.astype('int')
            if dataset_name in ['hellaswag', 'winogrande']:
                prompt_vote_pred = prompt_vote_pred.astype('str')
            label_key = 'label'
            if dataset_name in ['winogrande']:
                label_key = 'answer'
            elif dataset_name in ['story_cloze']:
                label_key = 'answer_right_ending'
            print(dev_datasets[label_key])
            pseudo_label_datasets = dev_datasets.remove_columns(label_key)
            pseudo_label_datasets = pseudo_label_datasets.add_column(label_key, prompt_vote_pred.tolist())
            pseudo_label_datasets = pseudo_label_datasets.add_column('confidence', confidence.tolist())
            print(pseudo_label_datasets[label_key])
            pseudo_label_datasets = pseudo_label_datasets.flatten_indices()
            pseudo_label_datasets = build_pseudo_dataset(dataset_name, dataset_config_name, pseudo_label_datasets, 32 * np.power(5, epoch + 1))
            all_train_dataset = get_prompted_datasets(dataset_name, dataset_config_name, args, accelerator, preprocess_train_function, preprocess_eval_function, column_names, pseudo_label_datasets)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{task_name}.bin'))



def do_train(args, model, tokenizer, train_dataset, raw_datasets, accelerator, dataset_name, dataset_config_name, epoch, template_list, preprocess_function):
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_train_collator = DataCollatorForMultipleChoiceTraining(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
        data_eval_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataset = train_dataset.shuffle(42)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_train_collator, batch_size=args.per_device_train_batch_size)
    train_dataloader = accelerator.prepare(train_dataloader)

    total_train_batch_size = args.per_device_train_batch_size * accelerator.num_processes
    total_dev_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    # optimizer_grouped_parameters = []
    # TODO: 设置一下
    # optimizer_grouped_parameters.append(
    #     {'params': [soft_prompt_embed.weight], 'weight_decay': 0.00001})

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.00001,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = Adafactor(optimizer_grouped_parameters,
                          lr=args.lr,
                          scale_parameter=False,
                          relative_step=False,
                          warmup_init=False,
                          weight_decay=0.00001)
    model.train()

    

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num train examples = {len(train_dataloader)}")
    # logger.info(f"  Num dev examples = {len(dev_dataloader)}")
    logger.info(f"  Instantaneous train batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Instantaneous dev batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_train_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_dev_batch_size}")

    progress_bar = tqdm(range(3), disable=not accelerator.is_local_main_process)

    global_step = 0
    train_losses = []
    best_eval_metric = 0
    best_task_summary = {}
    task_name = dataset_name + '_' + dataset_config_name if dataset_config_name is not None else dataset_name
    for _ in tqdm(range(3)):
        for batch in train_dataloader:
            global_step += 1

            model_inputs = {
                k: batch[k]
                for k in ["input_ids", "attention_mask", "labels"]
            }

            model_output = model(**model_inputs)

            logits = model_output.logits
            logits = torch.log_softmax(logits, dim=-1)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), model_inputs['labels'].view(-1))

            # loss = model_output.loss
            train_losses.append(loss.detach().cpu())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            model.zero_grad()
        progress_bar.update(1)
        #     if global_step == args.num_training_steps:
        #         break
        # if global_step == args.num_training_steps:
        #     break
    confidence, prompt_vote_pred = get_pseudo_label(args, dataset_name, dataset_config_name, template_list, model, tokenizer, raw_datasets, preprocess_function, accelerator, epoch + 1)
    return confidence, prompt_vote_pred


def do_eval(args, model, tokenizer, dev_dataset_dict, accelerator, data_eval_collator):
    # Metrics
    metric = load_metric("accuracy")
    template_result_summary = {}
    accuracy_sum = 0
    for uniq_task_name, dataset in dev_dataset_dict.items():
        if uniq_task_name.startswith('hellaswag'):
            task_name = 'hellaswag'
        elif uniq_task_name.startswith('super_glue_copa'):
            task_name = 'super_glue_copa'
        elif uniq_task_name.startswith('anli_r1'):
            task_name = 'anli_r1'
        elif uniq_task_name.startswith('anli_r2'):
            task_name = 'anli_r2'
        elif uniq_task_name.startswith('anli_r3'):
            task_name = 'anli_r3'
        elif uniq_task_name.startswith('super_glue_rte'):
            task_name = 'super_glue_rte'
        elif uniq_task_name.startswith('super_glue_cb'):
            task_name = 'super_glue_cb'
        elif uniq_task_name.startswith('super_glue_wsc.fixed'):
            task_name = 'super_glue_wsc.fixed'
        elif uniq_task_name.startswith('super_glue_wic'):
            task_name = 'super_glue_wic'
        elif uniq_task_name.startswith('winogrande_winogrande_xl'):
            task_name = 'winogrande_winogrande_xl'
        elif uniq_task_name.startswith('story_cloze'):
            task_name = 'story_cloze'

        # 只评测一次，加快计算
        if args.only_eval_single_template and task_name in template_result_summary:
            continue

        eval_dataloader = DataLoader(dataset, collate_fn=data_eval_collator, batch_size=args.per_device_eval_batch_size)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        print(f'evaluating task: {uniq_task_name}')
        for batch in eval_dataloader:
            model_inputs = {
                k: batch[k]
                for k in ["input_ids", "attention_mask", "labels"]
            }

            with torch.no_grad():
                # [batch_size, seq_len, vocab]
                logits = model(**model_inputs).logits

            # [batch_size, seq_len, vocab]
            masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
            # [batch_size, seq_len]
            seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
            # [batch_size, ]
            seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
            seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
            # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.

            # [batch_size, choice_num]
            predictions = seq_log_prob.argmax(dim=-1)

            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["targets"]),
            )

        eval_metric = metric.compute()
        accelerator.print(f"Result for {uniq_task_name}: {eval_metric}")
        accuracy_sum += eval_metric['accuracy']
        if task_name not in template_result_summary:
            template_result_summary[task_name] = []
        template_result_summary[task_name].append(eval_metric['accuracy'])

    result_summary = dict()
    for task_name, results in template_result_summary.items():
        result_summary[task_name] = sum(results)   # TODO: 为啥是sum

    result_num = 0   # 测了多少个task
    for k, v in template_result_summary.items():
        result_num += len(v)

    # 返回整体平均的acc， 以及每个数据集各个任务的sum acc
    return accuracy_sum / result_num, result_summary


def get_pseudo_label(args, dataset_name, dataset_config_name, template_list, model, tokenizer, raw_datasets, preprocess_function, accelerator, epoch):
    '''return pseudo label and confidence'''
    metric = load_metric("accuracy")
    prompts = DatasetTemplates(
            f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
            template_dir=args.template_dir)
    column_names = raw_datasets.column_names
    total_number = len(raw_datasets)
    task_name = f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}' 

    all_predictions = []
    all_log_probs = []
    all_logits = []
    ROOT_DIR = '/share/zongyu/chonghua/GPS_clean/T0'
    if epoch != 0:
        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            if template.metadata.original_task is not True:
                continue

            with accelerator.main_process_first():
                eval_dataset = raw_datasets.map(
                    preprocess_function, fn_kwargs={"template": template}, batched=True, remove_columns=column_names,
                    load_from_cache_file=False)
                eval_ids = eval_dataset['id']
                eval_dataset = eval_dataset.remove_columns(['id'])

            # DataLoaders creation:
            if args.pad_to_max_length:
                data_collator = default_data_collator
            else:
                data_collator = DataCollatorForMultipleChoice(
                    tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                )
            eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                        batch_size=args.per_device_eval_batch_size)
            eval_dataloader = accelerator.prepare(eval_dataloader)

            # Eval!
            total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

            logger.info("***** Running evaluation *****")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
            logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

            cur_all_predictions = []
            cur_log_probs = []
            cur_logits = []
            model.eval()
            for batch in eval_dataloader:
                # batch: attention_mask, input_ids, labels(答案的token ids), labels_attention_mask
                model_inputs = {
                    k: batch[k]
                    for k in ["input_ids", "attention_mask", "labels"]
                }
                with torch.no_grad():
                    # [batch_size, seq_len, vocab]
                    logits = model(**model_inputs).logits

                # [batch_size, seq_len, vocab]
                masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
                # [batch_size, seq_len]
                seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
                # [batch_size, ]
                seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
                seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
                max_log_prob = seq_log_prob.max(dim=-1)[0]
                # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
                # [batch_size, choice_num]
                predictions = seq_log_prob.argmax(dim=-1)

                cur_all_predictions.extend(predictions.cpu().numpy().tolist())
                cur_log_probs.extend(max_log_prob.cpu().numpy().tolist())
                cur_logits.extend(seq_log_prob.cpu().numpy().tolist())

                progress_bar.update(1)

            # copa
            filled_probs = np.ones(total_number) * -np.inf
            filled_predictions = np.ones(total_number) * -1
            filled_logits = np.zeros((total_number, num_labels_mapping[f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}']))
            for idx, prediction, log_prob, logits in zip(eval_ids, cur_all_predictions, cur_log_probs, cur_logits):
                filled_probs[idx] = log_prob
                filled_predictions[idx] = prediction
                filled_logits[idx] = logits
            filled_predictions = filled_predictions.tolist()
            filled_probs = filled_probs.tolist()
            filled_logits = filled_logits.tolist()
            all_predictions.append(filled_predictions)
            all_log_probs.append(filled_probs)
            all_logits.append(filled_logits)

    if accelerator.is_main_process:
        regularized_name = f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}_{dataset_config_name}"
        if epoch == 0:
            idx_and_preds = np.load(os.path.join(ROOT_DIR, f'{regularized_name}.npy'), allow_pickle=True)
            idx_and_preds = idx_and_preds.item()
            all_predictions = idx_and_preds['all_predictions']
            all_logits = idx_and_preds['all_logits']
        else:
            all_predictions = np.array(all_predictions)
            all_logits = np.array(all_logits)

        sorted_logits = np.sort(all_logits, axis=-1)
        confidence = np.exp(sorted_logits[..., -1]) - np.exp(sorted_logits[..., -2])
        sum_confidence = confidence.sum(axis=-1)
        sum_confidence = sum_confidence.reshape(len(sum_confidence), -1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(sum_confidence)
        kmeans_index = kmeans.cluster_centers_.argmax()
        prompt_idxes = np.where(kmeans.labels_ == kmeans_index)[0]
        # prompt_idxes = np.where(keep_sum_p_m_p_confidence >= np.mean(keep_sum_p_m_p_confidence))[0]
        keep_prompt_idx = np.array(list(range(TOP_K_for_each_task[task_name])))
        if len(prompt_idxes) < len(keep_prompt_idx) // 2:
            prompt_idxes = np.argsort(sum_confidence)[-len(keep_prompt_idx) // 2:]
        else:
            prompt_idxes = keep_prompt_idx[prompt_idxes]
        # filter bad new prompts
        prompt_vote_pred = all_logits[prompt_idxes].sum(axis=0).argmax(axis=-1)
        sum_prompt_confidence = confidence.sum(axis=0)

        return sum_prompt_confidence, prompt_vote_pred


if __name__ == "__main__":
    main()
