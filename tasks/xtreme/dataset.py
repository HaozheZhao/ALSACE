from ast import Raise
import json
from datasets import load_dataset, load_metric, concatenate_datasets
from datasets import Dataset, Features, ClassLabel
from itertools import combinations
from pandas.core.dtypes.common import validate_all_hashable
import random
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict
from random import randint, sample

task_to_keys = {
    "paws-x": ("sentence1", "sentence2"),
    
}

logger = logging.getLogger(__name__)

def sort_score(data_dict):
    # data_dict = json.load(open(gen_file_dir, 'r'))
    text_set = []
    new_data_dict = []
    for data in data_dict:
        text = data["text"] if "text" in data else data["text1"]
        if text not in text_set:
            new_data_dict.append(data)
        text_set.append(text)
    data_dict = new_data_dict
    scores = np.array([data["score"] for data in data_dict])
    sort_idx = np.argsort(-scores)
    new_dict = []
    for i in range(len(sort_idx)):
        new_dict.append(data_dict[sort_idx[i]])
    return new_dict
class XtremeDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config
        self.mask = self.tokenizer.mask_token
        self.pad = self.tokenizer.pad_token

        if data_args.dataset_name in ["paws-x"]:
            self.verbalizer_dict = {
                "2": {
                    # "0": {"0": "no", "1": "yes", "2": "yes"},
                    # "1": {"0": "no", "1": "yes", "2": "yes"},
                    # "2": {"0": "no", "1": "yes", "2": "yes"},
                    # "3": {"0": "no", "1": "yes", "2": "yes"},
                    # "4": {"0": "否", "1": "是", "2": "yes"},
                    # "5": {"0": "no", "1": "yes", "2": "yes"},
                    # "6": {"0": "no", "1": "yes", "2": "yes"},
                    "0": {"0": "no", "1": "yes"},
                    "1": {"0": "no", "1": "yes"},
                    "2": {"0": "no", "1": "yes"},
                    "3": {"0": "no", "1": "yes"},
                    "4": {"0": "no", "1": "yes"},
                    "5": {"0": "no", "1": "yes"},
                    "6": {"0": "no", "1": "yes"},
                    
                },
            }
            self.multilang_id_dict={
                "0":"en",
                "1":"de",
                "2":"es",
                "3":"fr",
                "4":"zh",
                "5":"ko",
                "6":"ja",
            }
            self.multilang_promote={
                "0":["Are the following sentences " ," and ", "  are paraphrases of each other? yes or no? Answer: "],
                }

            self.multilang_to_id =  dict(zip(self.multilang_id_dict.values(),self.multilang_id_dict.keys()))

        self.pre_seq_len = self.model_args.pre_seq_len
        self.raw_datasets = {}
        for key,value in self.multilang_id_dict.items():
            raw_data = load_dataset("specific-prompt-nodata/tasks/xtreme/xtreme_dataset.py",f"{data_args.dataset_name.upper()}.{value}", ignore_verifications=True) 
            self.raw_datasets[value] = raw_data

        self.label_list =list(set(self.raw_datasets['en']["train"]["label"]))
        self.num_labels = len(self.label_list)

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        self.label2token = {each :self.verbalizer_dict[str(self.num_labels)][each] for each in self.multilang_id_dict.keys()}
        self.token2label = { each:{v: k for k, v in self.verbalizer_dict[str(self.num_labels)][each].items()} for  each in self.multilang_id_dict.keys()}
        self.label_token_list ={  each: [v for _, v in self.verbalizer_dict[str(self.num_labels)][each].items()] for  each in self.multilang_id_dict.keys()}
        if self.config.model_type == 'mt5':
            self.label_token_ids_list ={each : [self.tokenizer.encode(l,max_length=4)[:-1] for l in self.label_token_list[each]] for  each in self.multilang_id_dict.keys()}
        else:
            self.label_token_ids_list ={each : [self.tokenizer.encode(l,max_length=4)[1:-1] for l in self.label_token_list[each]] for  each in self.multilang_id_dict.keys()}

        new_token_lists_all={}
        for multilang_id in self.multilang_id_dict.keys():
            new_token_lists=[]
            max_token_length = max([len(each) for each in self.label_token_ids_list[multilang_id]])
            for each in self.label_token_ids_list[multilang_id]:
                new_list =[0]*max_token_length # max_token of all == 2:
                new_list[:len(each)] = each
                new_token_lists.append(new_list)
            new_token_lists_all[multilang_id] =new_token_lists
        self.label_token_ids_list = new_token_lists_all

        self.label2id = {label: id for id, label in enumerate(self.label_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}
        print(f"{self.label2token}")
        print(f"{self.token2label}")
        print(f"{self.label_token_list}")
        print(f"{self.label_token_ids_list}")
        print(f"{self.label2id}")
        print(f"{self.id2label}")

        if self.data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        self.preprocess_function_one_list = {
            "paws-x": self.preprocess_function_one_PAWS
        }

        self.preprocess_function_two_list = {
            "paws-x": self.preprocess_function_two

        }

        if self.data_args.dataset_name == "paws-x":
            self.raw_datasets['en']["train"] = self.raw_datasets['en']["train"].map(lambda example: {"set_type": "train"})
            self.raw_datasets['en']["validation"] = self.raw_datasets['en']["validation"].map(lambda example: {"set_type": "en"})
            self.raw_datasets['en']["test"] = self.raw_datasets['en']["test"].map(lambda example: {"set_type": "en"})
            dataset_name = list(self.raw_datasets.keys())

            # self.raw_datasets[f"validation.{self.multilang_id_dict[self.model_args.verbalizer_id]}"] = self.raw_datasets[f"validation.{self.multilang_id_dict[self.model_args.verbalizer_id]}"].map(lambda example: {"set_type": "validation"})
            # self.raw_datasets[f"test.{self.multilang_id_dict[self.model_args.verbalizer_id]}"] = self.raw_datasets[f"test.{self.multilang_id_dict[self.model_args.verbalizer_id]}"].map(lambda example: {"set_type": "test"})
            for each in dataset_name:
                self.raw_datasets[each] = self.raw_datasets[each].map(
                    self.preprocess_function_one_list[self.data_args.dataset_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Runing one tokenization"
                )

                self.raw_datasets[each] = self.raw_datasets[each].map(
                    self.preprocess_function_two_list[self.data_args.dataset_name],
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Runing two tokenization"
                )
            dataset_name.remove('en')
            for each in dataset_name:
                self.raw_datasets[each] = self.raw_datasets[each].map(lambda example: {"set_type": each})
        if self.data_args.zero_tuning is not None:
            if training_args.do_train:
                # dataset_name = list(self.raw_datasets.keys())
                # dataset_name.remove('train')
                # test_dataset=self.raw_datasets["test.en"]
                # for each in dataset_name:
                #     if 'test' in each:
                #         _,lang=each.split('.')
                #         test_dataset = test_dataset.add_column(f"input_ids_{lang}",self.raw_datasets[each]['input_ids'])
                #         test_dataset = test_dataset.add_column(f"label_{lang}",self.raw_datasets[each]["label"])
                # self.train_dataset = test_dataset

                # to_lang = ['ara', 'bul', 'de', 'el', 'spa', 'fra',  'ru',  'th', 'vie', 'zh','en'] # xnli
                
                weak_lang = ['ja', 'ko', 'zh', 'de', 'fr']
                lang_rank = [ 'en', 'es','fr','de','zh','ko','ja'] # qqp
                to_lang = [ 'en', 'es','fr','de','zh','ko','ja'] # qqp
                # to_lang = ['en','de', 'fr', 'zh','ja','ko'] # qqp
                # to_lang = self.multilang_id_dict.values()
                if self.data_args.ensemble_lang_num and (len(to_lang)> self.data_args.ensemble_lang_num):
                    to_lang = to_lang[:self.data_args.ensemble_lang_num]
                    lang_pair = list(combinations(to_lang,2))

                elif self.data_args.ensemble_lang_pair:
                    top_lang=lang_rank[:self.data_args.ensemble_lang_pair]
                    weak_lang=lang_rank[-self.data_args.ensemble_lang_pair:]
                    if self.data_args.ensemble_lang_connect:
                        lang_pair=[]
                        for lang in weak_lang:
                            lang_pair+=list(combinations(top_lang+[lang],2))
                        lang_pair = list(set(lang_pair))
                    else:
                        lang_pair = list(combinations(top_lang+weak_lang,2))
                elif self.data_args.ensemble_weak_number and self.data_args.ensemble_lang_connect:
                    top_lang=lang_rank[:len(to_lang) - self.data_args.ensemble_weak_number]
                    weak_lang=lang_rank[-self.data_args.ensemble_weak_number:]
                    lang_pair=[]
                    for lang in weak_lang:
                        lang_pair+=list(combinations(top_lang+[lang],2))
                    lang_pair = list(set(lang_pair))

                else:
                    if self.data_args.ensemble_lang_connect:
                        lang_pair=[]
                        for lang in weak_lang:
                            lang_pair+=list(combinations(to_lang+[lang],2))
                        lang_pair = list(set(lang_pair))
                    else:
                        lang_pair = list(combinations(to_lang,2))
                if self.data_args.ensemble_lang is not None:
                    lang_pair = list(combinations(self.data_args.ensemble_lang.replace(' ','').split(','),2))
                lang_inputs1=[]
                lang_inputs2=[]
                lang_label1=[]
                lang_label2=[]
                types1=[]
                types2=[]

                lang_dict={}
                for each in to_lang:
                    with open(f'SuperGen/test_data_QQP/test.{each}.json', mode='r') as f:
                        lang_dict[each]= json.load(f)
           
                new_dataset={}
                for each in to_lang:
                    info = lang_dict[each]
                    for data in info:
                        data_dict={}
                        premise = data['text1']
                        hypothesis = data['text2']
                        data_dict['label']=int(data['label'])
                        # data_dict['label']=label_dict[data['label']]
                        if self.config.model_type =='mt5':
                            input_tokens =''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.multilang_promote["0"][2],'.']) 
                        else:
                            input_tokens = ''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis,self.multilang_promote["0"][2],self.mask, '.'])   
                        data_dict['input_ids'] = self.tokenizer(input_tokens, padding=False, max_length=512, truncation=True)["input_ids"]
                        data_dict['set_type'] = each
                        if each not in new_dataset:
                            new_dataset[each]=[]
                        new_dataset[each].append(data_dict)
                for lang in lang_pair:
                    inputs_0 = [each['input_ids'] for each in new_dataset[lang[0]]]
                    inputs_1 = [each['input_ids'] for each in new_dataset[lang[1]]]                    
                    
                    label_0 = [each['label'] for each in new_dataset[lang[0]]]
                    label_1 = [each['label'] for each in new_dataset[lang[1]]]                    
                    
                    set_type_0 = [each['set_type'] for each in new_dataset[lang[0]]]
                    set_type_1 = [each['set_type'] for each in new_dataset[lang[1]]]
      

                    lang_inputs1 = lang_inputs1 + inputs_0
                    lang_inputs2 = lang_inputs2 + inputs_1
                    lang_label1 = lang_label1 + label_0
                    lang_label2 = lang_label2 + label_1
                    types1 = types1 + set_type_0
                    types2 = types2 + set_type_1

                # for lang in lang_pair:
                #     lang_inputs1 = lang_inputs1 + self.raw_datasets[lang[0]]['validation']['input_ids'][:900]
                #     lang_inputs2 = lang_inputs2 + self.raw_datasets[lang[1]]['validation']['input_ids'][:900]
                #     lang_label1 = lang_label1 + self.raw_datasets[lang[0]]['validation']['label'][:900]
                #     lang_label2 = lang_label2 + self.raw_datasets[lang[1]]['validation']['label'][:900]
                #     types1 = types1 + self.raw_datasets[lang[0]]['validation']['set_type'][:900]
                #     types2 = types2 + self.raw_datasets[lang[1]]['validation']['set_type'][:900]

                
                dataset_dict ={"input_ids_1":lang_inputs1, "input_ids_2":lang_inputs2,"label_1":lang_label1 ,"label_2":lang_label2,"set_type1":types1 ,"set_type2":types2}
                ds=Dataset.from_dict(dataset_dict)
                ds =ds.shuffle()
                self.train_dataset = ds

                if data_args.max_train_samples is not None: # is None
                    self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
            if training_args.do_eval:
                self.eval_dataset = self.raw_datasets[self.multilang_id_dict[self.model_args.verbalizer_id]]['validation']
                if data_args.max_eval_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
            if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
                self.predict_dataset = self.raw_datasets[self.multilang_id_dict[self.model_args.verbalizer_id]]['test']
                if data_args.max_predict_samples is not None:
                    self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))       
        
        else:

            if training_args.do_train:
                if data_args.self_training is not None:
                    with open(f'self_training.json', mode='r') as f:
                        train_data= json.load(f)
                    label=[]
                    input_ids =[]
                    sentence_ids=[]
                    for data in train_data:
                        premise = data['sentence1']
                        hypothesis = data['sentence2']
                        if self.config.model_type =='mt5':
                            input_tokens =''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.multilang_promote["0"][2],'.']) 
                        else:
                            input_tokens = ''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis,self.multilang_promote["0"][2],self.mask, '.'])   
                        input_ids.append(self.tokenizer(input_tokens, padding=False, max_length=512, truncation=True)["input_ids"])
                        label.append(int(data['label']))
                        sentence_token =''.join([premise, hypothesis])
                        sentence_ids.append(self.tokenizer(sentence_token, padding=False, max_length=512, truncation=True)["input_ids"])
                    dataset_dict ={"input_ids":input_ids, "label":label,'sentence_ids':sentence_ids}
                    ds=Dataset.from_dict(dataset_dict)
                    ds =ds.shuffle()
                    self.train_dataset = ds
                else:
                    self.train_dataset = self.raw_datasets['en']["train"]
                    if data_args.max_train_samples is not None: # is None
                        self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
            if training_args.do_eval:
                self.eval_dataset = self.raw_datasets[self.multilang_id_dict[self.model_args.verbalizer_id]]['validation']
                if data_args.max_eval_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
            if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
                self.predict_dataset = self.raw_datasets[self.multilang_id_dict[self.model_args.verbalizer_id]]['test']
                if data_args.max_predict_samples is not None:
                    self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
        self.all_eval_dataset = {each: self.raw_datasets[each]['validation'] for each in list(self.raw_datasets.keys())  if 'validation' in self.raw_datasets[each] }

        self.all_test_dataset = { each: self.raw_datasets[each]['test'] for each in list(self.raw_datasets.keys()) if 'test' in self.raw_datasets[each] }

        self.metric = load_metric("specific-prompt-nodata/tasks/superglue/superglue_metric.py", data_args.dataset_name)
        self.test_key = "accuracy" if data_args.dataset_name not in ["record", "multirc"] else "f1"


    def data_collator(self, features):
        first = features[0]
        batch = {}

        # labels work
        # labels, label_token_ids_list -> batch
        if self.data_args.all_lang_prompt:
            lang_type = features[0]['set_type'].split('.')[1] if features[0]['set_type'] != 'train' else 'en'
        else:
            lang_type = 'en'
        pad_key_list = ["input_ids",  "labels"] if self.config.model_type in [ "t5",'mt5'] else ["input_ids", "sentence_ids", "labels"]
        for f in features:
            if self.config.model_type in ["gpt2", "t5",'mt5']: # t5和gpt2
                # f["labels"] = f["label_token_ids"]
                f["label_token_ids_list"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                if self.data_args.zero_tuning is not None and( "label_1" in f):
                        pad_key_list=[]
                        for lang in [1,2]:
                            with self.tokenizer.as_target_tokenizer():
                                label_token_ids = self.tokenizer(self.verbalizer_dict[str(self.num_labels)]["0"][str(f[f'label_{lang}'])], padding=False, max_length=512, truncation=True)["input_ids"]
                            f[f"labels_{lang}"] = label_token_ids
                            pad_key_list.append(f"input_ids_{lang}")
                            pad_key_list.append(f"labels_{lang}")               
            elif self.config.model_type in ["bert", "roberta", "albert", "deberta-v2","xlm-roberta"]: # 普通的模型
                if self.data_args.dataset_name in ['paws-x']:
                    if self.data_args.zero_tuning is not None and( "label_1" in f):
                        pad_key_list=[]
                        # for lang in self.multilang_to_id.keys():
                        for lang in [1,2]:
                            label_token_ids = self.label_token_ids_list[self.multilang_to_id[lang_type]][int(f[f"label_{lang}"])]
                            label_ids = [-100 for _ in range(len(f[f"input_ids_{lang}"]))]
                            mask_start = f[f"input_ids_{lang}"].index(self.tokenizer.mask_token_id)
                            label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
                            f[f"labels_{lang}"] = label_ids
                            f[f"label_token_ids_list_{lang}"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                            pad_key_list.append(f"input_ids_{lang}")
                            pad_key_list.append(f"labels_{lang}")
                    else:
                        label_token_ids = self.label_token_ids_list[self.multilang_to_id[lang_type]][int(f["label"])]
                        label_ids = [-100 for _ in range(len(f["input_ids"]))]
                        mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                        label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
                        f["labels"] = label_ids
                        f["label_token_ids_list"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                

        # Padding work
        #input_ids, sentence_ids, labels -> batch
        for key in pad_key_list:
            result = self.tokenizer.pad(
                {"input_ids": [f[key] for f in features]},
                padding=self.padding,
                max_length=self.max_seq_length,
                # pad_to_multiple_of=2,
                return_tensors="pt",
            )
            batch[key] = result["input_ids"]
            if self.config.model_type =='mt5':
                batch[f"{key}_attention_mask"] = result["attention_mask"]
            else:
                if self.data_args.zero_tuning is not None and( "label_1" in f):
                    batch[f"attention_mask_{key.split('_')[-1]}"] = result["attention_mask"]

                else: 
                    if key == "input_ids" and "attention_mask" not in batch.keys():
                        batch["attention_mask"] = result["attention_mask"]

        reduced_column = []
        reduced_column.extend(["input_ids", "sentence_ids", "attention_mask", "label_token_ids", "labels"]) # data_collator pad
        reduced_column.extend(["idx", "input_tokens", "sentence_tokens", "label_tokens"]) # preprocess_function_pre
        reduced_column.extend(["choice1_ids", "choice2_ids"]) # copa
        if self.data_args.zero_tuning is not None and( "label_1" in f):
            reduced_column.extend(pad_key_list)
        # reduced_column.extend(["label_token_ids_list"]) # xnli
        
        for k, v in f.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])
            
        return batch

    def preprocess_function_three(self, example):
        if self.data_args.all_lang_prompt:
            lang_type =  example['set_type'].split('.')[1] if example['set_type'] != 'train' else 'en'
        else:
            lang_type ='en'
        if self.data_args.dataset_name in ["boolq", "cb," "rte", "wic", "multirc",'nli','paws-x']:
            label_token_ids = self.label2token[self.multilang_to_id[lang_type]][str(example["label"])]
            label_ids = [-100 for _ in range(len(example["input_ids"]))]
            mask_start = example["input_ids"].index(self.tokenizer.mask_token_id)
            label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids

        example["labels"] = label_ids
        return example

    def preprocess_function_two(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        if self.config.model_type == 'mt5':
            with self.tokenizer.as_target_tokenizer():
                label_token_ids = self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"]
            result['labels'] = label_token_ids
        #     result = {
        #     "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        #     # "input_attention_mask": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["attention_mask"],
        #     "label_token_ids": self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        #     # "label_attention_mask": self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["attention_mask"],
        #     "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        #     # "sentence_attention_mask": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["attention_mask"],

        # }
        if self.config.model_type =='mt5':
            for each in result['input_ids'] :
                if len(each) == 512:
                    each[-1] = 1              
            for each in result['sentence_ids'] :
                if len(each) == 512:
                    each[-1] = 1            
        else:
            for each in result['input_ids'] :
                if len(each) == 512:
                    each[-2] =self.tokenizer.mask_token_id            
            for each in result['sentence_ids'] :
                if len(each) == 512:
                    each[-2] = self.tokenizer.mask_token_id

        return result


    def preprocess_function_one_PAWS(self, examples): # change the xglue into promote 
        key1,key2 = task_to_keys[self.data_args.dataset_name.lower()]
        sentence1 = examples[key1]
        sentence2 = examples[key2]
        result = {}

        # input_tokens
        if self.model_args.template_id == "2":
            if self.data_args.all_lang_prompt:
                lang_type = examples['set_type'].split('.')[1] if examples['set_type'] != 'train' else 'en'
            else:
                lang_type='en'
            if self.config.model_type =='mt5':
                result["input_tokens"] =''.join([self.multilang_promote[self.multilang_to_id[lang_type]][0],sentence1, self.multilang_promote[self.multilang_to_id[lang_type]][1],sentence2, self.multilang_promote[self.multilang_to_id[lang_type]][2],'.'])    
            else:
                result["input_tokens"] =''.join([self.multilang_promote[self.multilang_to_id[lang_type]][0],sentence1, self.multilang_promote[self.multilang_to_id[lang_type]][1],sentence2, self.multilang_promote[self.multilang_to_id[lang_type]][2],self.mask, '.'])    
        
        # sentence_tokens
        result["sentence_tokens"] = ''.join([sentence1, sentence2])

        # label_tokens
        result["label"] = int(examples["label"])
        
        result["label_tokens"] = self.label2token[self.multilang_to_id[lang_type]][str(examples["label"])]

        return result    


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

