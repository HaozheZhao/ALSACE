from ast import Raise
import json
from datasets import load_dataset, load_metric, concatenate_datasets
from datasets import Dataset, Features, ClassLabel
from itertools import combinations
from pandas.core.dtypes.common import validate_all_hashable
import random
import torch
from torchmetrics.text.rouge import ROUGEScore
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import tqdm
import numpy as np
import pandas as pd
from glob import glob
import logging
from collections import defaultdict
from random import randint, sample
from torchmetrics import BLEUScore

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "xnli": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
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
class XGlueDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})
        print(self.tokenizer.additional_special_tokens)
        self.prompt = '歌'
        self.mask = self.tokenizer.mask_token
        self.pad = self.tokenizer.pad_token
        self.cls = self.tokenizer.cls_token
        if data_args.dataset_name in ["xnli","cb", "rte"]:
            self.verbalizer_dict = {
                "2": {
                    "0": {"0": "yes", "1": "no", "-1": "no"},
                    "1": {"0": "true", "1": "false", "-1": "a"},
                },
                "3": {
                    "0": {"0": "yes", "1": "maybe", "2": "no"},
                    "1": {"0": "نعم", "1": "ربما", "2": "لا"},
                    "2": {"0": "Да", "1": "може би", "2": "Не"},
                    "3": {"0": "ja", "1": "vielleicht", "2": "Nein"},
                    "4": {"0": "Ναι", "1": "ίσως", "2": "Όχι"},
                    "5": {"0": "sí", "1": "quizás", "2": "no"},
                    "6": {"0": "oui", "1": "peut-être", "2": "Non"},
                    "7": {"0": "हाँ", "1": "शायद", "2": "नहीं"},
                    "8": {"0": "да", "1": "возможно", "2": "Нет"},
                    "9": {"0": "Ndiyo", "1": "Labda", "2": "La"},
                    "10": {"0": "ใช่", "1": "บางที", "2": "ไม่ใช่"},
                    "11": {"0": "evet", "1": "belki", "2": "Hayır"},
                    "12": {"0": "ہاں", "1": "شاید", "2": "نہيں"},
                    "13": {"0": "Có", "1": "có lẽ", "2": "Không"},
                    "14": {"0": "是", "1": "或许", "2": "否"}

                },
            }
            self.multilang_id_dict={
                "0":"en",
                "1":"ar",
                "2":"bg",
                "3":"de",
                "4":"el",
                "5":"es",
                "6":"fr",
                "7":"hi",
                "8":"ru",
                "9":"sw",
                "10":"th",
                "11":"tr",
                "12":"ur",
                "13":"vi",
                "14":"zh",
            }
            self.multilang_promote={
                "0":["Suppose " ," Can we infer that ", "? Yes, no or maybe? Answer: "],
                "1":["افترض "," هل يمكننا أن نستنتج أن  ","؟ نعم ، لا ، أو ربما؟ إجابه: "],
                "2":["Да предположим " ," Можем ли да заключим, че ", "? Да, не или може би? Отговор: "],
                "3":["Vermuten " ," Können wir darauf schließen ", "? Ja, nein oder vielleicht? Antworten: "],
                "4":["Υποθέτω " ," Μπορούμε να το συμπεράνουμε ", "? Ναι, όχι ή μήπως; Απάντηση: "],
                "5":["Suponer " ," ¿Podemos inferir que ", "? ¿Sí, no o tal vez? Responder: "],
                "6":["Supposer " ," Pouvons-nous en déduire que ", "? Oui, non ou peut-être ? Réponse: "],
                "7":["मान लीजिए " ," क्या हम इसका अनुमान लगा सकते हैं ", "? हाँ, नहीं या शायद? उत्तर: "],
                "8":["Предполагать " ," Можем ли мы сделать вывод, что ", "? Да, нет или может быть? Отвечать: "],
                "9":["Tuseme " ," Je, tunaweza kudokeza hilo ", "? Ndio, hapana au labda? Jibu: "],
                "10":["สมมติ " ," เราสามารถสรุปได้ว่า ", "? ใช่ ไม่ หรืออาจจะ? ตอบ: "],
                "11":["Sanmak " ," bunu çıkarabilir miyiz ", "? Evet, hayır ya da belki? Cevap: "],
                "12":["فرض کریں۔ " ," کیا ہم اس کا اندازہ لگا سکتے ہیں۔ ", "؟ ہاں، نہیں یا شاید؟ جواب:"],
                "13":["Giả sử " ," Chúng ta có thể suy luận rằng ", "? Có, không hoặc có thể? Câu trả lời: "],
                "14":["假设 ", " 我们可以推断出 ","? 是，不是，或者也许？ 答案: "]
                }
            self.label_dict= {'contradiction':2, 'entailment':0, 'neutral':1}
            self.label_dict_ver= {2:'no', 0:'yes', 1:'maybe'}
            
            self.multilang_to_id =  dict(zip(self.multilang_id_dict.values(),self.multilang_id_dict.keys()))


        self.pre_seq_len = self.model_args.pre_seq_len
        if data_args.dataset_name == 'mlama':
            train_data = pd.read_pickle('tasks/xtreme/mlama_data/train_en.pkl')
            test_file = glob("tasks/xtreme/mlama_data/test_"+'*.pkl')     
            valid_file = glob("tasks/xtreme/mlama_data/valid_"+'??.pkl')     
            lang_dict={
                "0":"en",
                "1":"ar",
                "2":"bg",
                "3":"de",
                "4":"el",
                "5":"es",
                "6":"fr",
                "7":"hi",
                "8":"ru",
                "9":"sw",
                "10":"th",
                "11":"tr",
                "12":"ur",
                "13":"vi",
                "14":"zh",
                "15":"fr",
            }
            self.raw_datasets={}
            self.raw_datasets['train'] = Dataset.from_pandas(train_data)
            self.raw_datasets['test'] = {}
            self.raw_datasets['validation'] ={}
            for file in test_file:
                lang = file.split('/')[-1].replace('.pkl','').replace('test_','')
                if lang in lang_dict.values():
                    self.raw_datasets['test'][lang] = Dataset.from_pandas(pd.read_pickle(file))

            for file in valid_file:
                lang = file.split('/')[-1].replace('.pkl','').replace('valid_','')
                if lang in lang_dict.values():
                    self.raw_datasets['validation'][lang] = Dataset.from_pandas(pd.read_pickle(file))
            if self.data_args.pad_to_max_length:
                self.padding = "max_length"
            else:
                self.padding = "longest"

        elif data_args.dataset_name == 'geolama':
            train_data = pd.read_pickle('GeoLama.pkl')
            test_file = pd.read_pickle("GeoLama_test.pkl")     
            valid_file = pd.read_pickle("GeoLama_test.pkl")
            self.raw_datasets={}
            self.raw_datasets['train'] = Dataset.from_dict(train_data).shuffle()
            self.raw_datasets['test'] = Dataset.from_dict(test_file)
            self.raw_datasets['validation'] =Dataset.from_dict(test_file)
            if self.data_args.pad_to_max_length:
                self.padding = "max_length"
            else:
                self.padding = "longest"
       


        else:
            self.raw_datasets = load_dataset("tasks/xglue/xglue_dataset.py", data_args.dataset_name) 

            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)

            if self.data_args.pad_to_max_length:
                self.padding = "max_length"
            else:
                self.padding = "longest"

            self.label2token = {each :self.verbalizer_dict[str(self.num_labels)][each] for each in self.multilang_id_dict.keys()}
            self.token2label = { each:{v: k for k, v in self.verbalizer_dict[str(self.num_labels)][each].items()} for  each in self.multilang_id_dict.keys()}
            self.label_token_list ={  each: [v for _, v in self.verbalizer_dict[str(self.num_labels)][each].items()] for  each in self.multilang_id_dict.keys()}
            if self.config.model_type in ['mt5','xglm']:
                self.label_token_ids_list ={each : [[self.tokenizer.encode(l,max_length=4)[:-1][-1]] for l in self.label_token_list[each]] for  each in self.multilang_id_dict.keys()}
            else:
                self.label_token_ids_list ={each : [self.tokenizer.encode(l,max_length=4)[1:-1] for l in self.label_token_list[each]] for  each in self.multilang_id_dict.keys()}


        # self.label2token = self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id]
        # self.token2label = {v: k for k, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()}
        # self.label_token_list = [v for _, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()]
        # self.label_token_ids_list = [self.tokenizer.encode(l,max_length=4)[1:-1] for l in self.label_token_list]
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
            "boolq": self.preprocess_function_one_boolq,
            "rte": self.preprocess_function_one_nli,
            "cb": self.preprocess_function_one_nli,
            "wic": self.preprocess_function_one_wic,
            "wsc": self.preprocess_function_one_wsc,
            "copa": self.preprocess_function_one_copa,
            "multirc": self.preprocess_function_one_multirc,
            "xnli": self.preprocess_function_one_nli,
            "mlama": self.preprocess_function_one_mlama,
            "geolama": self.preprocess_function_one_geolama
        }

        self.preprocess_function_two_list = {
            "boolq": self.preprocess_function_two,
            "rte": self.preprocess_function_two,
            "cb": self.preprocess_function_two,
            "wic": self.preprocess_function_two,
            "wsc": self.preprocess_function_two_wsc,
            "copa": self.preprocess_function_two_copa,
            "multirc": self.preprocess_function_two,
            "xnli": self.preprocess_function_two,
            "mlama": self.preprocess_function_two_mlama,
            "geolama": self.preprocess_function_two_geolama,


        }

        if self.data_args.dataset_name == "wsc":
            self.raw_datasets["train"] = self.raw_datasets["train"].map(lambda example: {"set_type": "train"}).filter(lambda example: example['label'] == 1)
            self.raw_datasets["validation"] = self.raw_datasets["validation"].map(lambda example: {"set_type": "validation"})
            self.raw_datasets["test"] = self.raw_datasets["validation"].map(lambda example: {"set_type": "test"})
        elif self.data_args.dataset_name == "copa":
            train_len = len(self.raw_datasets["train"])
            updated_dataset =  self.raw_datasets["train"].map(lambda example: {'idx': example['idx'] + train_len, 'choice1': example['choice2'], 'choice2': example["choice1"], 'label': 1 - example['label']})
            self.raw_datasets["train"] = concatenate_datasets([self.raw_datasets["train"], updated_dataset])
        elif self.data_args.dataset_name == "xnli":
            # dict(zip(self.multilang_id_dict.values(),self.multilang_id_dict.keys()))
            self.raw_datasets["train"] = self.raw_datasets["train"].map(lambda example: {"set_type": "train"})
            dataset_name = list(self.raw_datasets.keys())
            dataset_name.remove('train')
            for each in dataset_name:
                self.raw_datasets[each] = self.raw_datasets[each].map(lambda example: {"set_type": each})
        elif self.data_args.dataset_name in ["mlama","geolama"]:
            self.raw_datasets["train"] = self.raw_datasets["train"].map(lambda example: {"set_type": "train"})
            # self.raw_datasets[f"validation.{self.multilang_id_dict[self.model_args.verbalizer_id]}"] = self.raw_datasets[f"validation.{self.multilang_id_dict[self.model_args.verbalizer_id]}"].map(lambda example: {"set_type": "validation"})
            # self.raw_datasets[f"test.{self.multilang_id_dict[self.model_args.verbalizer_id]}"] = self.raw_datasets[f"test.{self.multilang_id_dict[self.model_args.verbalizer_id]}"].map(lambda example: {"set_type": "test"})
        if data_args.dataset_name in ["mlama","geolama"]:      
            self.raw_datasets['train'] = self.raw_datasets['train'].map(
            self.preprocess_function_one_list[self.data_args.dataset_name],
            load_from_cache_file= data_args.overwrite_cache,
            desc="Runing one tokenization"
                )

            self.raw_datasets['train'] = self.raw_datasets['train'].map(
                self.preprocess_function_two_list[self.data_args.dataset_name],
                batched=True,
                load_from_cache_file= data_args.overwrite_cache,
                desc="Runing two tokenization"
            )
            if data_args.dataset_name == 'mlama':
                for each in self.raw_datasets['test']:
                    self.raw_datasets['test'][each] = self.raw_datasets['test'][each].map(
                    self.preprocess_function_one_list[self.data_args.dataset_name],
                    load_from_cache_file= data_args.overwrite_cache,
                    desc="Runing one tokenization"
                        )

                    self.raw_datasets['test'][each] = self.raw_datasets['test'][each].map(
                        self.preprocess_function_two_list[self.data_args.dataset_name],
                        batched=True,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Runing two tokenization"
                    )
                for each in self.raw_datasets['validation']:
                    self.raw_datasets['validation'][each] = self.raw_datasets['validation'][each].map(
                    self.preprocess_function_one_list[self.data_args.dataset_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Runing one tokenization"
                        )

                    self.raw_datasets['validation'][each] = self.raw_datasets['validation'][each].map(
                        self.preprocess_function_two_list[self.data_args.dataset_name],
                        batched=True,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Runing two tokenization"
                    )
            else:
                self.raw_datasets['test'] = self.raw_datasets['test'].map(
                self.preprocess_function_one_list[self.data_args.dataset_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Runing one tokenization"
                    )
                self.raw_datasets['test'] = self.raw_datasets['test'].map(
                    self.preprocess_function_two_list[self.data_args.dataset_name],
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Runing two tokenization"
                )
                self.raw_datasets['validation'] = self.raw_datasets['validation'].map(
                self.preprocess_function_one_list[self.data_args.dataset_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Runing one tokenization"
                    )

                self.raw_datasets['validation'] = self.raw_datasets['validation'].map(
                    self.preprocess_function_two_list[self.data_args.dataset_name],
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Runing two tokenization"
                )
        else:
            self.raw_datasets = self.raw_datasets.map(
                self.preprocess_function_one_list[self.data_args.dataset_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Runing one tokenization"
            )

            self.raw_datasets = self.raw_datasets.map(
                self.preprocess_function_two_list[self.data_args.dataset_name],
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Runing two tokenization"
            )

        if 1 == 0: # Only works in multi-task setting
            self.data_collator = DataCollatorWithPadding
            self.raw_datasets = self.raw_datasets.map(
                self.preprocess_function_three,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Runing three tokenization"
            )
        
        if self.data_args.zero_tuning is not None:
            if training_args.do_train:
                to_lang = ['ar', 'th','bg', 'de', 'el', 'es', 'fr',  'ru',  'vi', 'zh','en','sw','ur','tr','hi']

                lang_rank = ['en', 'fr', 'bg', 'es', 'de', 'vi', 'ru', 'el', 'zh', 'th', 'tr','hi', 'ar', 'ur', 'sw']

                weak_lang =[ 'tr','hi', 'ar', 'ur', 'sw']

                # weak_lang = ['fa', 'sw']
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
                    with open(f'SuperGen/test_data/test.{each}.json', mode='r') as f:
                        lang_dict[each]= json.load(f)

                new_dataset={}
                for each in to_lang:
                    info = lang_dict[each]
                    for data in info:
                        data_dict={}
                        premise = data['text1']
                        hypothesis = data['text2']
                        data_dict['label']=self.label_dict[data['label']]
                        if self.config.model_type in ['mt5']:
                            input_tokens =''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.multilang_promote["0"][2],'.'])  
                        elif self.config.model_type in ['xglm']:
                            input_tokens =''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.multilang_promote["0"][2],self.cls])  
                        else:
                            input_tokens = ''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.prompt * self.pre_seq_len,self.multilang_promote["0"][2],self.mask, '.'])   
                        data_dict['input_ids'] = self.tokenizer(input_tokens, padding=False, max_length=512, truncation=True)["input_ids"]
                        data_dict['set_type'] = each
                        if each not in new_dataset:
                            new_dataset[each]=[]
                        new_dataset[each].append(data_dict)

                
                for lang in lang_pair:
                    lang_inputs1 = lang_inputs1 + self.raw_datasets[f'validation.{lang[0]}']['input_ids'][:500]
                    lang_inputs2 = lang_inputs2 + self.raw_datasets[f'validation.{lang[1]}']['input_ids'][:500]
                    lang_label1 = lang_label1 + self.raw_datasets[f'validation.{lang[0]}']['label'][:500]
                    lang_label2 = lang_label2 + self.raw_datasets[f'validation.{lang[1]}']['label'][:500]
                    types1 = types1 + self.raw_datasets[f'validation.{lang[1]}']['set_type'][:500]
                    types2 = types2 + self.raw_datasets[f'validation.{lang[1]}']['set_type'][:500]
                
                dataset_dict ={"input_ids_1":lang_inputs1, "input_ids_2":lang_inputs2,"label_1":lang_label1 ,"label_2":lang_label2,"set_type1":types1 ,"set_type2":types2}
                ds=Dataset.from_dict(dataset_dict)
                ds =ds.shuffle()
                self.train_dataset = ds

                if data_args.max_train_samples is not None: # is None
                    self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
            if training_args.do_eval:
                self.eval_dataset = self.raw_datasets[f"validation.{self.multilang_id_dict[self.model_args.verbalizer_id]}"]
                if data_args.max_eval_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
            if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
                self.predict_dataset = self.raw_datasets[f"test.{self.multilang_id_dict[self.model_args.verbalizer_id]}"]
                if data_args.max_predict_samples is not None:
                    self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))            
        
        else:

            if training_args.do_train:
                if self.data_args.generate_train  is not None:

                    with open('SuperGen/data_test/MNLI/train.json', mode='r') as f:
                        train_data= json.load(f)
                    # train_data = sort_score(train_data)
                    if self.data_args.max_train_samples is not None:
                        train_data = sample(train_data,self.data_args.max_train_samples)

                    label=[]
                    input_ids =[]
                    sentence_ids=[]
                    for data in train_data:
                        premise = data['text1']
                        hypothesis = data['text2']
                        label.append(self.label_dict[data['label']])
                        input_tokens = ''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.prompt * self.pre_seq_len,self.multilang_promote["0"][2],self.mask, '.'])   
                        input_ids.append(self.tokenizer(input_tokens, padding=False, max_length=512, truncation=True)["input_ids"])
                        sentence_token =''.join([premise, hypothesis])
                        sentence_ids.append(self.tokenizer(sentence_token, padding=False, max_length=512, truncation=True)["input_ids"])
                    dataset_dict ={"input_ids":input_ids, "label":label,'sentence_ids':sentence_ids}
                    ds=Dataset.from_dict(dataset_dict)
                    ds =ds.shuffle()
                    self.train_dataset = ds
                elif self.data_args.self_training is not None:
                    with open('tasks/xtreme/mt5_2epoch_xnli_self_training_data.json', mode='r') as f:
                        train_data= json.load(f)
                    # train_data = train_data[-569:]
                    # model_train_data = self.raw_datasets["train"]
                    # if self.data_args.max_train_samples is not None:
                    #     model_train_data = sample(model_train_data,self.data_args.max_train_samples)
                    label=[]
                    input_ids =[]
                    sentence_ids=[]
                    for data in train_data:
                        premise = data['text1']
                        hypothesis = data['text2']
                        label.append(data['label'])
                        if self.config.model_type =='mt5':
                            input_tokens = ''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis,self.multilang_promote["0"][2], '.'])  
                        else:
                            input_tokens = ''.join([self.multilang_promote["0"][0],premise, self.multilang_promote["0"][1],hypothesis, self.prompt * self.pre_seq_len,self.multilang_promote["0"][2],self.mask, '.'])   
                        input_ids.append(self.tokenizer(input_tokens, padding=False, max_length=512, truncation=True)["input_ids"])
                        sentence_token =''.join([premise, hypothesis])
                        sentence_ids.append(self.tokenizer(sentence_token, padding=False, max_length=512, truncation=True)["input_ids"])
                    # input_ids = input_ids + model_train_data["input_ids"] 
                    # label = label + model_train_data["label"] 
                    # sentence_ids = sentence_ids + model_train_data["sentence_ids"] 
                    dataset_dict ={"input_ids":input_ids, "label":label,'sentence_ids':sentence_ids}
                    ds=Dataset.from_dict(dataset_dict)
                    ds =ds.shuffle()
                    self.train_dataset = ds

                else:
                    self.train_dataset = self.raw_datasets["train"]
                    if data_args.max_train_samples is not None: # is None
                        self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
            if training_args.do_eval:
                if self.data_args.dataset_name == "mlama":
                    self.eval_dataset = self.raw_datasets['validation']["zh"]
                elif self.data_args.dataset_name == "geolama":
                    self.eval_dataset = self.raw_datasets['validation']
                else:
                    self.eval_dataset = self.raw_datasets[f"validation.{self.multilang_id_dict[self.model_args.verbalizer_id]}"]

                if data_args.max_eval_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
            if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
                if self.data_args.dataset_name == "mlama":
                    self.eval_dataset = self.raw_datasets['validation']["zh"]
                elif self.data_args.dataset_name == "geolama":
                    self.eval_dataset = self.raw_datasets['validation']
                else:
                    self.predict_dataset = self.raw_datasets[f"test.{self.multilang_id_dict[self.model_args.verbalizer_id]}"]
                if data_args.max_predict_samples is not None:
                    self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
        
        if self.data_args.dataset_name == "mlama":
            
            self.all_eval_dataset = { valid:self.raw_datasets['validation'][valid] for valid in self.raw_datasets['validation']}
            self.all_test_dataset = { test:self.raw_datasets['test'][test] for test in self.raw_datasets['test']}
        
        else:
        
            validate_dataset = [ each for each in list(self.raw_datasets.keys()) if 'validation' in each ]

            test_dataset = [ each for each in list(self.raw_datasets.keys()) if 'test' in each ]


            self.all_eval_dataset = { valid:self.raw_datasets[valid] for valid in validate_dataset}
            self.all_test_dataset = { test:self.raw_datasets[test] for test in test_dataset}


        self.metric = load_metric("tasks/superglue/superglue_metric.py", data_args.dataset_name)
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
        if self.config.model_type in [ "t5",'mt5']:
            pad_key_list = ["input_ids",  "labels"] 
        elif self.config.model_type in [ 'xglm']:
            pad_key_list = ["input_ids",  "labels",'label_ids'] 
        else:
            pad_key_list = ["input_ids", "sentence_ids", "labels"]
        
        for f in features:
            if self.config.model_type in ["gpt2", "t5",'mt5','xglm']: # t5和gpt2

                if self.data_args.zero_tuning is not None and( "label_1" in f):
                        pad_key_list=[]
                        for lang in [1,2]:
                            with self.tokenizer.as_target_tokenizer():
                                label_token_ids = self.tokenizer(self.label_dict_ver[f[f'label_{lang}']], padding=False, max_length=512, truncation=True)["input_ids"]
                            f[f"labels_{lang}"] = label_token_ids
                            pad_key_list.append(f"input_ids_{lang}")
                            pad_key_list.append(f"labels_{lang}")
                        f["label_token_ids_list"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                else:
                    with self.tokenizer.as_target_tokenizer():
                        label_token_ids = self.tokenizer(self.label_dict_ver[f['label']], padding=False, max_length=512, truncation=True)["input_ids"]
                    f["labels"] = label_token_ids
                    f["label_token_ids_list"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                    
            elif self.config.model_type in ["bert", "roberta", "albert", "deberta-v2","xlm-roberta",'xlm','xlm-roberta-xl']: # 普通的模型
                if self.data_args.dataset_name in ["boolq", "cb", "rte", "wic", "multirc",'xnli']:
                    if self.data_args.zero_tuning is not None and( "label_1" in f):
                        pad_key_list=[]
                        # for lang in self.multilang_to_id.keys():
                        for lang in [1,2]:
                            label_token_ids = self.label_token_ids_list[self.multilang_to_id[lang_type]][f[f"label_{lang}"]]
                            label_ids = [-100 for _ in range(len(f[f"input_ids_{lang}"]))]
                            mask_start = f[f"input_ids_{lang}"].index(self.tokenizer.mask_token_id)
                            label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
                            f[f"labels_{lang}"] = label_ids
                            f[f"label_token_ids_list_{lang}"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                            pad_key_list.append(f"input_ids_{lang}")
                            pad_key_list.append(f"labels_{lang}")
                    else:
                        label_token_ids = self.label_token_ids_list[self.multilang_to_id[lang_type]][f["label"]]
                        label_ids = [-100 for _ in range(len(f["input_ids"]))]
                        mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                        label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
                        f["labels"] = label_ids
                        f["label_token_ids_list"] = self.label_token_ids_list[self.multilang_to_id[lang_type]]
                elif self.data_args.dataset_name in ["mlama","geolama"]:
                    label_ids = [-100 for _ in range(len(f["input_ids"]))]

                    mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                    try:
                        label_pad_idx = f["labels"].index(self.tokenizer.pad_token_id)
                    except:
                        label_pad_idx = len(f['labels'])
                    if self.data_args.dataset_name =="geolama":
                        mask_pad = len(f["labels"][:label_pad_idx])
                        f["input_ids"] = f["input_ids"][:mask_start]+ [self.tokenizer.mask_token_id]*mask_pad+f["input_ids"][mask_start+1:]
                        label_ids = [-100 for _ in range(len(f["input_ids"]))]

                    label_ids[mask_start: mask_start + len(f["labels"][:label_pad_idx])] = f["labels"][:label_pad_idx]
                    f["labels"] = label_ids

                
                elif self.data_args.dataset_name in ["wsc"]: 
                    label_ids = [-100 for _ in range(len(f["input_ids"]))]
                    mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                    label_ids[mask_start: mask_start + len(f["label_token_ids"][1:-1])] = f["label_token_ids"][1:-1]
                    f["labels"] = label_ids
                elif self.data_args.dataset_name in ["copa"]: 
                    label_ids = [-100 for _ in range(len(f["input_ids"]))]
                    mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                    label_ids[mask_start: mask_start + len(f["label_token_ids"][1:-1])] = f["label_token_ids"][1:-1]
                    f["labels"] = label_ids
                    for choice in ["choice1", "choice2"]:
                        mask_end = mask_start + len(f[f'{choice}_ids'][1:-1])
                        label_ids = [-100 for _ in range(len(f["input_ids"]))]
                        label_ids[mask_start: mask_end] = f[f'{choice}_ids'][1:-1]
                        f[f'{choice}_ids'] = label_ids

        if self.data_args.dataset_name == "copa":
            pad_key_list.extend(["choice1_ids", "choice2_ids"])
        if self.data_args.dataset_name =='geolama':
            pad_key_list.remove('sentence_ids')
        for key in pad_key_list:
            result = self.tokenizer.pad(
                {"input_ids": [f[key] for f in features]},
                padding=self.padding,
                max_length=self.max_seq_length,
                # pad_to_multiple_of=2,
                return_tensors="pt",
            )
            batch[key] = result["input_ids"]
            if self.config.model_type in ['mt5','xglm']:
                batch[f"{key}_attention_mask"] = result["attention_mask"]
            else:
                if self.data_args.zero_tuning is not None and( "label_1" in f):
                    batch[f"attention_mask_{key.split('_')[-1]}"] = result["attention_mask"]

                else: 
                    if key == "input_ids" and "attention_mask" not in batch.keys():
                        batch["attention_mask"] = result["attention_mask"]

        reduced_column = []
        reduced_column.extend(["input_ids", "sentence_ids", "attention_mask", "label_token_ids", "labels", "label_ids"]) # data_collator pad
        reduced_column.extend(["idx", "input_tokens", "sentence_tokens", "label_tokens"]) # preprocess_function_pre
        reduced_column.extend(["choice1_ids", "choice2_ids"]) # copa
        if self.data_args.zero_tuning is not None and( "label_1" in f):
            reduced_column.extend(pad_key_list)
        
        for k, v in f.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])

        # WSC thing
        if self.data_args.dataset_name == "wsc":
            batch["label_token_ids"] = [f["label_token_ids"] for f in features]
        
        if self.data_args.dataset_name == "geolama":
            batch['opition'] = [f['opition'] for f in features]

        return batch

    def preprocess_function_three(self, example):
        if self.data_args.all_lang_prompt:
            lang_type =  example['set_type'].split('.')[1] if example['set_type'] != 'train' else 'en'
        else:
            lang_type ='en'
        if self.data_args.dataset_name in ["boolq", "cb," "rte", "wic", "multirc",'nli']:
            label_token_ids = self.label2token[self.multilang_to_id[lang_type]][str(example["label"])]
            label_ids = [-100 for _ in range(len(example["input_ids"]))]
            mask_start = example["input_ids"].index(self.tokenizer.mask_token_id)
            label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
        elif self.data_args.dataset_name in ["wsc", "copa"]: # 不需要 label_token_ids_list，需要label_token_ids
            label_ids = [-100 for _ in range(len(example["input_ids"]))]
            mask_start = example["input_ids"].index(self.tokenizer.mask_token_id)
            label_ids[mask_start: mask_start + len(example["label_token_ids"][1:-1])] = example["label_token_ids"][1:-1]

        example["labels"] = label_ids
        return example

    def preprocess_function_two_nli(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
      


        return result
    def preprocess_function_two(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        if self.config.model_type in ['mt5','xglm']:
            with self.tokenizer.as_target_tokenizer():
                label_token_ids = self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"]
            result['labels'] = label_token_ids
            result['label_ids'] = result['input_ids']

   

        return result

    def preprocess_function_two_mlama(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "label_tokens":  self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "labels":  self.tokenizer(examples["label"], padding='max_length', max_length=32, truncation=True,add_special_tokens=False)["input_ids"]
        }

        return result
    def preprocess_function_two_geolama(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "opition":  [self.tokenizer(each, padding=False, max_length=512, truncation=True)["input_ids"] for each in examples["opitions"]],
            "labels":  self.tokenizer(examples["label_tokens"], padding='max_length', max_length=32, truncation=True,add_special_tokens=False)["input_ids"],
            "sentence_ids" : self.tokenizer(examples["sentences"], padding='max_length', max_length=512, truncation=True,add_special_tokens=False)["input_ids"],
        }

        return result

    def preprocess_function_two_wsc(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "label_token_ids": self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        return result


    def preprocess_function_two_copa(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "label_token_ids": self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "choice1_ids": self.tokenizer(examples["choice1_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "choice2_ids": self.tokenizer(examples["choice2_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        return result


    def preprocess_function_one_boolq(self, examples):
        passage = examples["passage"][:450]
        question = examples["question"]
        result = {}

        # input_tokens
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([passage, question, self.prompt * self.pre_seq_len, self.mask])
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([passage, '. Question: ', question, self.prompt * self.pre_seq_len, '? Answer: ', self.mask, '.'])
        elif self.model_args.template_id == "2":
            result["input_tokens"] = ''.join([passage, '. Based on the previous passage, ', question, self.prompt * self.pre_seq_len, '?', self.mask, '.'])
        elif self.model_args.template_id == "3":
            result["input_tokens"] = ''.join(['Based on the following passage, ', question, self.prompt * self.pre_seq_len, '?', self.mask, '.', passage])
        elif self.model_args.template_id == "4": # No mask, for t5/gpt-2
            result["input_tokens"] =  ''.join([passage, question, self.prompt * self.pre_seq_len, "? Answer: "])
        else:
            raise NotImplementedError(
                "The template id {} has not been defined.".format(self.model_args.template_id)
            )

        # sentence_tokens
        result["sentence_tokens"] = ''.join([question])

        # label_tokens
        result["label"] = examples["label"]
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        elif self.config.model_type == "gpt2":
            result["label_tokens"] = ''.join([result["input_tokens"], result["label_tokens"]])

        return result          


    def preprocess_function_one_nli(self, examples): # change the xglue into promote 
        premise = examples["premise"]
        hypothesis = examples["hypothesis"]
        result = {}

        # input_tokens
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([premise, self.prompt * self.pre_seq_len, self.mask, hypothesis])
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([premise, '. Question: ', hypothesis, self.prompt * self.pre_seq_len, '? Answer: ', self.mask, '.'])
        elif self.model_args.template_id == "2":
            if self.data_args.all_lang_prompt:
                lang_type = examples['set_type'].split('.')[1] if examples['set_type'] != 'train' else 'en'
            else:
                lang_type='en'
            if self.config.model_type in ['mt5']:
                result["input_tokens"] =''.join([self.multilang_promote[self.multilang_to_id[lang_type]][0],premise, self.multilang_promote[self.multilang_to_id[lang_type]][1],hypothesis, self.multilang_promote[self.multilang_to_id[lang_type]][2],'.'])    

            elif self.config.model_type in ['xglm']:
                result["input_tokens"] =''.join([self.multilang_promote[self.multilang_to_id[lang_type]][0],premise, self.multilang_promote[self.multilang_to_id[lang_type]][1],hypothesis, self.multilang_promote[self.multilang_to_id[lang_type]][2],self.cls,'.'])    
            else:
                result["input_tokens"] =''.join([self.multilang_promote[self.multilang_to_id[lang_type]][0],premise, self.multilang_promote[self.multilang_to_id[lang_type]][1],hypothesis, self.prompt * self.pre_seq_len,self.multilang_promote[self.multilang_to_id[lang_type]][2],self.mask, '.'])    
        
        # sentence_tokens
        result["sentence_tokens"] = ''.join([premise, hypothesis])

        # label_tokens
        result["label"] = examples["label"]
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        elif  self.config.model_type in ['mt5','xglm']:
            result["label_tokens"] = self.label2token[self.multilang_to_id[lang_type]][str(examples["label"])]

        elif self.config.model_type == "gpt2":
            result["label_tokens"] = ''.join([result["input_tokens"], result["label_tokens"]])

        return result    
    def preprocess_function_one_mlama(self, examples): # change the xglue into promote 
        X = examples["sub_label"]
        Y = examples["obj_label"]
        template = examples['template'].replace('[Y]',self.mask*5)
        result = {}
        result["input_tokens"] = template.replace('[X]',X)

        result["sentence_tokens"] = examples['template'].replace('[Y]',Y).replace('[X]',X)

        # label_tokens
        result["label_tokens"] = examples['template'].replace('[X]',X).replace('[Y]',Y)
        result["label"] = Y

        return result    
    def preprocess_function_one_geolama(self, examples): # change the xglue into promote 
        inputs = examples["input"]
        label = examples["label"]
        opition = examples["opition"]
       
        result = {}
        result["input_tokens"] = inputs

        result["opitions"] = [inputs.replace("<mask>",i) for i in opition]
        # label_tokens
        result["label_tokens"] = label

        result['sentences'] =inputs.replace("<mask>",label)

        return result   
        
    def preprocess_function_one_wic(self, examples):
        sentence1 = examples["sentence1"]
        sentence2 = examples["sentence2"]
        word = examples["word"]
        result = {}

        # input_tokens
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([sentence1, word, self.prompt * self.pre_seq_len, self.mask, sentence2])
        elif self.model_args.template_id == "2":
            result["input_tokens"] =  ''.join(['"', sentence1, '" / "', sentence2, self.prompt * self.pre_seq_len, '" Similar sense of "', word, '"?', self.mask, '.'])
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([sentence1, sentence2, 'Does ' + word + ' have the same meaning in both sentences?', self.prompt * self.pre_seq_len, self.mask])
        elif self.model_args.template_id == "3":
            result["input_tokens"] =  ''.join([word, ' . Sense (1) (a) "', sentence1, '" (', self.mask, ') "', sentence2, '"'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([sentence1, sentence2])

        # label_tokens
        result["label"] = examples["label"]
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        elif self.config.model_type == "gpt2":
            result["label_tokens"] = ''.join([result["input_tokens"], result["label_tokens"]])

        return result     


    def preprocess_function_one_wsc(self, examples):
        text = examples["text"]
        span1_text = examples["span1_text"]
        span2_text = examples["span2_text"]
        num_pad = self.rng.randint(0, 3) if examples["set_type"] == "train" else 1
        masks = self.mask * (len(self.tokenizer(span1_text, padding=self.padding, max_length=512, truncation=True)["input_ids"][1: -1]) + num_pad)
        result = {}
        result["num_pad"] = num_pad

        # input_tokens
        if self.model_args.template_id == "0":
            pass
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([text, self.prompt * self.pre_seq_len, "The pronoun '*", span2_text + "*' refers to", masks, '.'])
        elif self.model_args.template_id == "2":
            result["input_tokens"] =  ''.join([text, self.prompt * self.pre_seq_len, "In the previous sentence, the pronoun '*", span2_text, "*' refers to", masks, '.'])
        elif self.model_args.template_id == "3":
            result["input_tokens"] =  ''.join([text, self.prompt * self.pre_seq_len, "Question: In the passage above, what does the pronoun '*", span2_text, "*' refer to? Answer: ", self.masks + '.'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([text])

        # label_tokens
        result["label"] = examples["label"]
        result["label_tokens"] = span1_text

        return result   


    def preprocess_function_one_copa(self, examples):
        premise = examples["premise"]
        question = examples["question"]
        choice1 = examples["choice1"]
        choice2 = examples["choice2"]
        num_masks = max(len(self.tokenizer(choice, padding=self.padding, max_length=512, truncation=True)["input_ids"][1: -1]) for choice in [choice1, choice2])
        result = {}

        # input_tokens

        if question == 'cause':
            joiner = "because"
            if self.model_args.template_id == "0":
                result["input_tokens"] =  ''.join(['"', choice1, '" or "', choice2, '"?', premise, joiner, self.prompt * self.pre_seq_len, self.mask * num_masks, '.'])
            elif self.model_args.template_id == "1":
                result["input_tokens"] =  ''.join([choice1, 'or', choice2, '?', premise, self.prompt * self.pre_seq_len, joiner, self.mask * num_masks, '.'])
        else:
            joiner = "so"
            if self.model_args.template_id == "0":
                result["input_tokens"] =  ''.join(['"', choice1, '" or "', choice2, '"?', premise, ', ', joiner, self.prompt * self.pre_seq_len, self.mask * num_masks, '.'])
            elif self.model_args.template_id == "1":
                result["input_tokens"] =  ''.join([choice1, 'or', choice2, '?', premise, ', ', joiner, self.prompt * self.pre_seq_len, self.mask * num_masks, '.'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([joiner])

        # label_tokens
        result["label"] = examples["label"]
        result["choice1_tokens"] = choice1
        result["choice2_tokens"] = choice2
        result["label_tokens"] = [choice1, choice2][result["label"]] # Only used in multi-task setting

        return result   


    def preprocess_function_one_multirc(self, examples):
        paragraph = examples["paragraph"][:450]
        question = examples["question"]
        answer = examples["answer"]
        if self.data_args.all_lang_prompt:
            lang_type = examples['set_type'].split('.')[1] if examples['set_type'] != 'train' else 'en'
        else:
            lang_type = 'en'

        result = {}

        # input_tokens
        if self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([paragraph, '. Question: ', question, self.prompt * self.pre_seq_len, '? Is it ', answer, '?', self.mask, '.'])
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([paragraph, '. Question: ', question, self.prompt * self.pre_seq_len, '? Is the correct answer "', answer, '"?', self.mask, '.'])
        if self.model_args.template_id == "2":
            result["input_tokens"] =  ''.join([paragraph, '. Based on the previous passage, ', question, self.prompt * self.pre_seq_len, '? Is "', answer, '" a correct answer?', self.mask, '.'])
        if self.model_args.template_id == "3":
            result["input_tokens"] =  ''.join([paragraph, question, self.prompt * self.pre_seq_len, '- [', self.mask, ']', answer])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([question])

        # label_tokens
        result["label"] = examples["label"]
        result["label_tokens"] = self.label2token[self.multilang_to_id[lang_type]][str(examples["label"])]

        return result   

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if self.data_args.dataset_name == 'mlama':
            preds[preds==-100] = 0
            bleu = BLEUScore()
            bleu_result = []
            predictions = self.tokenizer.batch_decode(preds,skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(p.label_ids,skip_special_tokens=True)
            count =0
            for idx, each in enumerate(predictions):
                if labels[idx] in each:
                    count+=1
                bleu_result.append(bleu([each],[[labels[idx]]]).item())
            return {"accuracy": count/len(labels),"avg_bleu":np.array(bleu_result).mean()}
        elif self.data_args.dataset_name == 'geolama':
            predictions = self.tokenizer.batch_decode(preds.argmax(-1),skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(p.label_ids,skip_special_tokens=True)
            count =0
            for idx, each in enumerate(predictions):
                if labels[idx] in each:
                    count+=1
            return {"accuracy": count/len(labels)}



        preds = np.argmax(preds, axis=1)
        if self.data_args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.data_args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

