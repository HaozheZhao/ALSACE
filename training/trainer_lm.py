import math

import torch
import torch.nn as nn
import time
import numpy as np
import logging
import os
import torch.nn.functional as F

from typing import Dict, OrderedDict, Union, Any, Optional, List, Tuple
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.functional import pad

from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.file_utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

from tasks.xtreme.task_helpers import TASK_HELPERS as xtreme_TASK_HELPERS
from tasks.xglue.task_helpers import TASK_HELPERS as xglue_TASK_HELPERS
from utils.embedding_encoder import PromptEncoder, EmbeddingEncoder

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

class LMTrainer(Trainer):
    def __init__(self, *args, model_args, data_args, predict_dataset = None, test_key = "accuracy",all_test_dataset=None,all_eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)

        #TODO 这个trainer类是可以拥有data_args, model_args, training_args和config的。
        # training_args就是self.args
        self.model_args = model_args
        self.data_args = data_args
        self.config = self.model.config

        if self.data_args.task_name == "xglue":
            self.task_helper = xglue_TASK_HELPERS[self.data_args.dataset_name](model_args=self.model_args, model=self.model, tokenizer=self.tokenizer)
        elif self.data_args.task_name == "xtreme":
            self.task_helper = xtreme_TASK_HELPERS[self.data_args.dataset_name](model_args=self.model_args, model=self.model, tokenizer=self.tokenizer)

        self.embedding_encoder = EmbeddingEncoder(self.config, self.model_args, self.model)
        if self.place_model_on_device:
            self._move_model_to_device(self.embedding_encoder, self.args.device)

        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        self.best_model=None

        self.all_eval_dataset = all_eval_dataset
        self.all_test_dataset =all_test_dataset
        
        self.label_names = ["label"]

    
    def get_KL_loss(self,logit1,logit2,log_target=True):
        kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=log_target)
        # input should be a distribution in the log space
        input = F.log_softmax(logit1, dim=1)
        # Sample a batch of distributions. Usually this would come from the dataset
        target = F.log_softmax(logit2, dim=1) if log_target else F.softmax(logit2, dim=1)
        output = kl_loss(input, target)
        return output


    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # if (epoch % 50 != 0) or epoch ==0 :
        #     return

        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            if "train" in self.model_args.eval_type:
                logger.info(f"***** Running Evaluation for train dataset *****")
                train_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval, eval_dataset=self.train_dataset)
                self._report_to_hp_search(trial, epoch, train_metrics)
            if "test" in self.model_args.eval_type:
                logger.info(f"***** Running Evaluation for test dataset *****")
                train_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval, eval_dataset=self.test_dataset)
                self._report_to_hp_search(trial, epoch, train_metrics)
            else:
                logger.info(f"***** Running Evaluation for eval dataset *****")    
                eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics[0]["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics[0]["eval_"+self.test_key]
                self.best_model = model
                self._save_checkpoint(self.best_model , trial, metrics=eval_metrics[0])

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            if eval_metrics is None:
                self._save_checkpoint(model, trial)
            else:
                self._save_checkpoint(model, trial, metrics=eval_metrics[0])
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]], step=None) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.

        We will conly keep inputs_embeds, labels and attention_mask in our input to model.
        """

        inputs = self._prepare_input(inputs)
        if self.model_args.prompt_type == "soft":
            inputs["inputs_embeds"] = self.embedding_encoder.id2embedding(inputs["input_ids"], inputs["sentence_ids"])
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        input_models = {}
        if self.model_args.prompt_type == "soft":
            for key in ["inputs_embeds", "attention_mask"]:
                input_models[key] = inputs[key]
        else:
            if self.config.model_type in ['mt5']:
                input_list=["input_ids", "input_ids_attention_mask","labels","labels_attention_mask","label_token_ids_list"]
                if self.data_args.zero_tuning is not None and( "label_1" in inputs):
                    input_list = ['input_ids_1', 'input_ids_1_attention_mask', 'labels_1', 'input_ids_2', 'input_ids_2_attention_mask', 'labels_2','label_token_ids_list']
                    input_models["decoder_input_ids_1"]=self.model.prepare_decoder_input_ids_from_labels(inputs['labels_1'])
                    input_models["decoder_input_ids_2"]=self.model.prepare_decoder_input_ids_from_labels(inputs['labels_2'])
                else:
                    input_models["decoder_input_ids"]=self.model.prepare_decoder_input_ids_from_labels(inputs['labels'])
            elif self.config.model_type in ['xglm']:
                input_list=["input_ids", "input_ids_attention_mask","labels","labels_attention_mask","label_token_ids_list",'label_ids']
                # eos_index = torch.where(inputs['input_ids'] == self.tokenizer.cls_token_id)[1]
                # for idx, end_idx in enumerate(eos_index):
                #     inputs['label_ids'][idx][end_idx:end_idx+len(inputs['labels'][idx][1:])] = inputs['labels'][idx][1:]


                if self.data_args.zero_tuning is not None and( "label_1" in inputs):
                    input_list = ['input_ids_1', 'input_ids_1_attention_mask', 'labels_1', 'input_ids_2', 'input_ids_2_attention_mask', 'labels_2','label_token_ids_list']
                else:
                    inputs['label_ids'][:,-1] = inputs['labels'][:,1]
                
                    ignore_index = torch.argmax((inputs['input_ids'] != self.tokenizer.eos_token_id).type(torch.int),dim=1)
                    for idx, ign_idx in enumerate(ignore_index):
                        inputs['label_ids'][idx][:ign_idx] = -100
            else:
                input_list = ["input_ids", "attention_mask","labels"]
                if self.data_args.zero_tuning is not None and( "label_1" in inputs):
                    input_list = ['input_ids_1', 'attention_mask_1', 'labels_1', 'input_ids_2', 'attention_mask_2', 'labels_2','label_token_ids_list_1','label_token_ids_list_2']
            for key in input_list:
                input_models[key] = inputs[key]
        if self.config.model_type in ['mt5']:
            if self.data_args.zero_tuning is not None and( "label_1" in inputs):
                eos_index1 = torch.where(input_models['labels_1'] == self.tokenizer.eos_token_id)[1]
                eos_index2= torch.where(input_models['labels_2'] == self.tokenizer.eos_token_id)[1]
                for idx, end_idx in enumerate(eos_index1):
                    input_models['labels_1'][idx][end_idx+1:] = -100
                for idx, end_idx in enumerate(eos_index2):
                    input_models['labels_2'][idx][end_idx+1:] = -100
            else:
                eos_index = torch.where(input_models['labels'] == self.tokenizer.eos_token_id)[1]
                for idx, end_idx in enumerate(eos_index):
                    input_models['labels'][idx][end_idx+1:] = -100
        if self.data_args.dataset_name and  self.data_args.zero_tuning is None:
            pad_mask = torch.where(input_models['labels'] == self.tokenizer.pad_token_id)
            input_models['labels'][pad_mask] = -100
        return inputs, input_models

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        _, input_models = self._prepare_inputs(inputs, step="train")

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, input_models, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        if self.config.model_type in ['mt5','t5']:
            loss, outputs = self.compute_loss(model, input_models, return_outputs=True, step="train")
            if self.data_args.dataset_name in ["copa", "record"]:
                loss = self.task_helper.logits2loss(inputs, outputs)
        else:
            with self.autocast_smart_context_manager():
                loss, outputs = self.compute_loss(model, input_models, return_outputs=True, step="train")
                if self.data_args.dataset_name in ["copa", "record"]:
                    loss = self.task_helper.logits2loss(inputs, outputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        #TODO 这里把用来计算acc的东西存下来
        inputs, input_models = self._prepare_inputs(inputs, step="eval")
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, input_models)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    # with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, input_models, return_outputs=True, step="eval")
                    if self.config.model_type == 'mt5':
                        poss = input_models['label_token_ids_list'][0].reshape(-1)
                        poss[1] =  1432
                        logits = outputs['scores'][0][:,poss]
                        # logits = self.task_helper.logits2pred(input_models,outputs['scores'][0])
                        # logits = outputs['scores'][0]
                        return (loss, logits, labels)
                    else:
                        # m=torch.argmax(logits,dim=1)
                        # result = [poss[each] for each in m]
                        # loss = loss.mean().detach()
                        # count =0
                        # for i ,each in enumerate(result):
                        #     if self.tokenizer.decode(each) in self.tokenizer.decode(input_models['decoder_input_ids'][i]):
                        #         count+=1
                        # print(count/len(input_models['decoder_input_ids']))

                        if isinstance(outputs, dict):
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                        else:
                            logits = outputs[1:]

                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        if self.data_args.dataset_name =='mlama':
                            outputs = model(inputs['input_ids'], inputs['attention_mask'],labels = inputs['labels'])
                            logits = self.task_helper.logits2pred(inputs, outputs.logits)
                            # loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                            # loss = loss_fn(outputs.logits,inputs['labels'])
                            # logits = torch.argmax(logits,dim=1)
                            las=[]
                            max_length = 5
                            for idx,i in enumerate(inputs["labels"]):
                                la = i[torch.where(i!=-100)[0]]
                                pad_length = max_length-len(la)
                                la = pad(la,(0,pad_length))
                                las.append(la)
                            return (outputs.loss, logits,torch.stack(las))
                        elif self.data_args.dataset_name =='geolama':
                            outputs = model(inputs['input_ids'], inputs['attention_mask'],labels = inputs['labels'])
                            las=[]
                            max_length = 5
                            for idx,i in enumerate(inputs["labels"]):
                                la = i[torch.where(i!=-100)[0]]
                                pad_length = max_length-len(la)
                                la = pad(la,(0,pad_length))
                                las.append(la)
                            return (outputs.loss, outputs.logits,torch.stack(las))


                        outputs = model(**input_models)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]


        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        if type(logits) == tuple and len(logits[1].shape)>1:
            logits = logits[1]
        elif type(logits) == tuple:
            logits = logits[0]
      
            
        logits = self.task_helper.logits2pred(inputs, logits)

        return (loss, logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, step=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.config.model_type in ['mt5']:
            if self.data_args.zero_tuning is not None and step =='train':
                from undecorated import undecorated
                from types import MethodType

                generate_with_grad = undecorated(model.generate)
                model.generate_with_grad = MethodType(generate_with_grad, model)
                input_ids1= inputs['input_ids_1']
                input_ids_attention_mask1=inputs['input_ids_1_attention_mask']
                # labels_ids1= None if self.label_smoother is not None else inputs['labels_1']                
                input_ids2= inputs['input_ids_2']
                input_ids_attention_mask2=inputs['input_ids_2_attention_mask']
                # labels_ids2= None if self.label_smoother is not None else inputs['labels_2']
                outputs1 = model.generate_with_grad(
                input_ids = input_ids1,
                attention_mask=input_ids_attention_mask1,
                max_length=2,
                output_scores =True,return_dict_in_generate =True
                )                
                outputs2 = model.generate_with_grad(
                input_ids = input_ids2,
                attention_mask=input_ids_attention_mask2,
                max_length=2,
                output_scores =True,return_dict_in_generate =True
                )

                inputs1={'input_ids':input_ids1,'label_token_ids_list':inputs['label_token_ids_list']}
                inputs2={'input_ids':input_ids2,'label_token_ids_list':inputs['label_token_ids_list']}
                # poss = inputs['label_token_ids_list'][0].reshape(-1)
                # logits1 = outputs1['scores'][0][:,poss]
                # logits2 = outputs2['scores'][0][:,poss]

                logits1 = self.task_helper.logits2pred(inputs1, outputs1.scores[0])
                logits2 = self.task_helper.logits2pred(inputs2, outputs2.scores[0])
                # loss = JSD()(logits1,logits2)
                loss = self.get_KL_loss(logits1,logits2)
                outputs = (outputs1,outputs2)
                return (loss, outputs) if return_outputs else loss
            else:
                input_ids= inputs['input_ids']
                input_ids_attention_mask=inputs['input_ids_attention_mask']
                labels_ids= None if self.label_smoother is not None else inputs['labels']
                decoder_input_ids = inputs['decoder_input_ids']
            if step =='train':
                outputs = model(input_ids = input_ids, attention_mask = input_ids_attention_mask, decoder_input_ids=decoder_input_ids , labels=labels_ids) 
            else:
                outputs = model.generate(
                input_ids = input_ids,
                attention_mask=input_ids_attention_mask,
                max_length=2,
                output_scores =True,return_dict_in_generate =True
                )
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(outputs.scores[0].view(-1, outputs.scores[0].size(-1)), labels_ids[:,0].view(-1))
                return (loss, outputs)

        elif self.config.model_type in ['xglm']:
            if self.data_args.zero_tuning is not None and step =='train':
                input_ids1= inputs['input_ids_1']
                input_ids_1_attention_mask=inputs['input_ids_1_attention_mask']
                input_ids2= inputs['input_ids_2']
                input_ids_2_attention_mask=inputs['input_ids_2_attention_mask']
                outputs1 = model(input_ids = input_ids1, attention_mask = input_ids_1_attention_mask)
                outputs2 = model(input_ids = input_ids2, attention_mask = input_ids_2_attention_mask)
                logits1 = self.task_helper.logits2pred(inputs1, outputs1.logits)
                logits2 = self.task_helper.logits2pred(inputs2, outputs2.logits)
                loss = self.get_KL_loss(logits1,logits2)
                outputs = (outputs1,outputs2)
                return (loss, outputs) if return_outputs else loss
            else:
                input_ids= inputs['input_ids']
                input_ids_attention_mask=inputs['input_ids_attention_mask']
                label_ids=inputs['label_ids']
                outputs = model(input_ids = input_ids, attention_mask = input_ids_attention_mask,labels = label_ids)
                

        else:
            if self.data_args.zero_tuning is not None and step =='train':
                input_ids1= inputs['input_ids_1']
                input_ids_attention_mask1=inputs['attention_mask_1']
                # labels_ids1= None if self.label_smoother is not None else inputs['labels_1']                
                input_ids2= inputs['input_ids_2']
                input_ids_attention_mask2=inputs['attention_mask_2']
                # labels_ids2= None if self.label_smoother is not None else inputs['labels_2']
                outputs1 = model(input_ids = input_ids1, attention_mask = input_ids_attention_mask1) 
                outputs2 = model(input_ids = input_ids2, attention_mask = input_ids_attention_mask2) 
                inputs1={'input_ids':input_ids1,'label_token_ids_list':inputs['label_token_ids_list_1']}
                inputs2={'input_ids':input_ids2,'label_token_ids_list':inputs['label_token_ids_list_2']}
                # if "xlm-roberta-xl" in model.config.model_type:
                # logits1 = self.get_roberta_xl(inputs1, outputs1.logits)
                # logits2 = self.get_roberta_xl(inputs2, outputs2.logits)
                # else:
                logits1 = self.task_helper.logits2pred(inputs1, outputs1.logits)
                logits2 = self.task_helper.logits2pred(inputs2, outputs2.logits)
                # logits1 = outputs1.logits
                # logits2 = outputs2.logits
                # answer_mask1 = torch.nonzero(inputs1["input_ids"] == self.model_args.tokenizer.mask_token_id)
                # logits1 = logits1[answer_mask1[:,0],answer_mask1[:,1]]
                # answer_mask2 = torch.nonzero(inputs2["input_ids"] == self.model_args.tokenizer.mask_token_id)
                # logits2 = logits2[answer_mask2[:,0],answer_mask2[:,1]]

                # loss = JSD()(logits1,logits2)
                loss = self.get_KL_loss(logits1,logits2)
                outputs = (outputs1,outputs2)
                return (loss, outputs) if return_outputs else loss

            else:
                if self.data_args.dataset_name=='mlama':
                    outputs = model(inputs['input_ids'], inputs['attention_mask'],labels = inputs['labels'])
                    # logits = self.task_helper.logits2pred(inputs, outputs.logits)
                    # loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    # loss = loss_fn(logits,inputs['labels'])
                    return (outputs.loss, outputs)
                else:
                    outputs = model(**inputs)
            
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss
    def get_roberta_xl(self,inputs, logits):
            num_mask_token = len(inputs["label_token_ids_list"][0][0])

            batch_size = inputs["input_ids"].shape[0]
            label_mask = torch.nonzero(inputs["input_ids"] == self.tokenizer.mask_token_id).reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.model.device) # batch_size * num_mask
    
            vocab_size = logits.shape[-1] # vocab_size
            index = label_mask.unsqueeze(2).repeat(1, 1, vocab_size).long() # batch_size * 1 * vocab_size
    
            y_pred = torch.gather(logits, index=index, dim=1)
            return y_pred.squeeze(1)
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return (output.metrics,output.predictions,output.label_ids)
    def save_logit(self, split, logit, name=None):

        path = os.path.join(self.args.output_dir, f"{split}_{name}.npy")
        np.save(path,logit)

def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    return result

