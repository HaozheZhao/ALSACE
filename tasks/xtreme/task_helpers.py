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

import math
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional, Any
import torch
import re
import numpy as np
from torch.nn import CrossEntropyLoss


class TaskHelper(ABC):
    """
    A helper class that provides custom training and evaluation methods for tasks that do not fit in PETs default
    schema, for example because they require more than two sequences of text, different evaluation metrics or
    verbalizers consisting of multiple tokens.
    """

    def __init__(self, model_args):
        """
        Create a new task helper.
        :param wrapper: The wrapper for the language model being used.
        """
        self.model_args = model_args
        self.output = None

    def logits2pred(self, inputs, logits):
        pass


class GeneralTaskHelper(TaskHelper):
    """
    A helper class that provides custom training and evaluation methods for tasks that do not fit in PETs default
    schema, for example because they require more than two sequences of text, different evaluation metrics or
    verbalizers consisting of multiple tokens.

    This task_halper is ok for BoolQ, RTE, CB, WiC
    WSC, ReCoRD, MultiRC and COPA requires specific task helper.
    """

    def __init__(self, model_args, model, tokenizer):
        """
        Create a new task helper.
        :param wrapper: The wrapper for the language model being used.
        """
        self.model = model
        self.config = self.model.config
        self.tokenizer = tokenizer
        self.model_args = model_args

    def logits2pred(self, inputs, logits):
        num_mask_token = len(inputs["label_token_ids_list"][0][0])

        batch_size = inputs["input_ids"].shape[0]

        if self.config.model_type in ["t5","mt5"]:
            num_of_token=len(inputs["label_token_ids_list"][0])
            logits = logits.unsqueeze(1).repeat(1,num_of_token,1)
            y_pred = torch.gather(logits, index=inputs["label_token_ids_list"], dim=2) # batch_size  * num_mask_token * num_labels
            y_pred_result = y_pred[:, :, 0].clone() # batch_size * num_labels
            for i in range(1, num_mask_token):
                y_pred_result *= y_pred[:, :, i]
            return y_pred_result
        elif self.config.model_type == "gpt2":
            logits = logits[:, -num_mask_token:, :]
            y_pred = torch.gather(logits, index=inputs["label_token_ids_list"].transpose(1, 2), dim=2) # batch_size  * num_mask_token * num_labels
            # y_pred = torch.gather(y_pred, index=inputs["label_token_id_list"], dim=1) # batch_size  * num_labels
            y_pred = y_pred.transpose(1, 2)  # batch_size  * num_labels * num_mask_token
            y_pred_result = y_pred[:, :, 0] # batch_size * num_labels
            for i in range(1, num_mask_token):
                y_pred_result *= y_pred[:, :, i]
            return y_pred_result
        else:
            # label_mask = (input_ids == self.model_args.tokenizer.mask_token_id).nonzero().reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
            label_mask = torch.nonzero(inputs["input_ids"] == self.model_args.tokenizer.mask_token_id).reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.model.device) # batch_size * num_mask
            
            
            vocab_size = logits.shape[-1] # vocab_size
            index = label_mask.unsqueeze(2).repeat(1, 1, vocab_size).long() # batch_size * 1 * vocab_size
            y_pred = torch.gather(logits, index=index, dim=1) # batch_size * num_mask_token * vocab_size
            y_pred = torch.gather(y_pred, index=inputs["label_token_ids_list"].transpose(1, 2), dim=2) # batch_size  * num_mask_token * num_labels
            # y_pred = torch.gather(y_pred, index=inputs["label_token_id_list"], dim=1) # batch_size  * num_labels
            y_pred = y_pred.transpose(1, 2)  # batch_size  * num_labels * num_mask_token
            y_pred_result = y_pred[:, :, 0].clone() # batch_size * num_labels
            for i in range(1, num_mask_token):
                y_pred_result *= y_pred[:, :, i]
            return y_pred_result



class WscTaskHelper(TaskHelper):
    """A custom task helper for the Wsc dataset."""

    def __init__(self, model_args, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    def logits2pred(self, inputs, logits):

        batch_size = inputs['input_ids'].shape[0]

        mask_positions = []
        for batch_id in range(batch_size):
            mask_positions.append([
                idx for idx, input_id in enumerate(inputs["input_ids"][batch_id]) if input_id == self.tokenizer.mask_token_id
            ])

        output_input_ids = torch.argmax(logits, dim=2)

        result = []
        for batch_id in range(batch_size):
            mask_output = [output_input_ids[batch_id][idx] for idx in mask_positions[batch_id]]
            if all(x in mask_output for x in inputs["label_token_ids"][batch_id]) or all(x in inputs["label_token_ids"][batch_id] for x in mask_output):
                result.append([0, 1])
            else:
                result.append([1, 0])

        return torch.tensor(result)

class CopaTaskHelper(TaskHelper):
    """
    A helper class that provides custom training and evaluation methods for tasks that do not fit in PETs default
    schema, for example because they require more than two sequences of text, different evaluation metrics or
    verbalizers consisting of multiple tokens.

    This task_halper is ok for BoolQ, RTE, CB, WiC
    WSC, ReCoRD, MultiRC and COPA requires specific task helper.
    """

    def __init__(self, model_args, model, tokenizer):
        """
        Create a new task helper.
        :param wrapper: The wrapper for the language model being used.
        """
        self.model = model
        self.config = self.model.config
        self.tokenizer = tokenizer
        self.model_args = model_args

    def logits2pred(self, inputs, logits):

        batch_size = inputs['input_ids'].shape[0]

        log_probabilities = []
        
        for batch_id in range(batch_size):
            masks_choice1 = [(idx, token_id) for idx, token_id in enumerate(inputs['choice1_ids'][batch_id]) if token_id != -100]
            masks_choice2 = [(idx, token_id) for idx, token_id in enumerate(inputs['choice2_ids'][batch_id]) if token_id != -100]

            log_probability = []

            # method1
            for mask_list in [masks_choice1, masks_choice2]:
                max_prob = None
                for m_pos, m_id in mask_list:
                    m_prob = logits[batch_id][m_pos][m_id].item()
                    if max_prob is None or m_prob > max_prob:
                        max_prob = m_prob
                if max_prob < 0: max_prob = 1e-5
                log_probability.append(math.log(max_prob))

            # method2
            # for mask_list in [masks_choice1, masks_choice2]:
            #     max_prob = 0
            #     prob_len = len(mask_list)
            #     for m_pos, m_id in mask_list:
            #         m_prob = logits[batch_id][m_pos][m_id].item()
            #         max_prob += m_prob
            #     max_prob /= prob_len
            #     log_probability.append(max_prob)

            log_probabilities.append(log_probability)
        
        return torch.Tensor(log_probabilities)

    def logits2loss(self, inputs, output):
        mask = inputs['label'].unsqueeze(1)
        correct_targets = inputs['choice1_ids'] * (1 - mask) + inputs['choice2_ids'] * mask
        wrong_targets = inputs['choice1_ids'] * mask + inputs['choice2_ids'] * (1 - mask)

        loss_fct = CrossEntropyLoss()
        loss_correct_label = loss_fct(output["logits"].view(-1, self.config.vocab_size), correct_targets.view(-1).to(self.model.device))
        loss_wrong_label = loss_fct(output["logits"].view(-1, self.config.vocab_size), wrong_targets.view(-1).to(self.model.device))
        loss = 1 + loss_correct_label - loss_wrong_label
        loss[loss < 0] = 0
        return loss


class RecordTaskHelper(TaskHelper):
    """
    A helper class that provides custom training and evaluation methods for tasks that do not fit in PETs default
    schema, for example because they require more than two sequences of text, different evaluation metrics or
    verbalizers consisting of multiple tokens.

    This task_halper is ok for BoolQ, RTE, CB, WiC
    WSC, ReCoRD, MultiRC and COPA requires specific task helper.
    """

    def __init__(self, model_args, model, tokenizer):
        """
        Create a new task helper.
        :param wrapper: The wrapper for the language model being used.
        """
        self.model = model
        self.config = self.model.config
        self.tokenizer = tokenizer
        self.model_args = model_args

    def logits2pred(self, inputs, logits):

        batch_size = inputs['input_ids'].shape[0]

        log_probabilities = []
        
        for batch_id in range(batch_size):
            masks_choice1 = [(idx, token_id) for idx, token_id in enumerate(inputs['choice1_token_ids'][batch_id]) if token_id != -100]
            masks_choice2 = [(idx, token_id) for idx, token_id in enumerate(inputs['choice2_token_ids'][batch_id]) if token_id != -100]

            log_probability = []

            # method1
            # for mask_list in [masks_choice1, masks_choice2]:
            #     max_prob = None
            #     for m_pos, m_id in mask_list:
            #         m_prob = logits[batch_id][m_pos][m_id].item()
            #         if max_prob is None or m_prob > max_prob:
            #             max_prob = m_prob
            #     log_probability.append(math.log(max_prob))

            # method2
            for mask_list in [masks_choice1, masks_choice2]:
                max_prob = 0
                prob_len = len(mask_list)
                for m_pos, m_id in mask_list:
                    m_prob = logits[batch_id][m_pos][m_id].item()
                    max_prob += m_prob
                max_prob /= prob_len
                log_probability.append(max_prob)

            log_probabilities.append(log_probability)
        
        return torch.Tensor(log_probabilities)

    def logits2loss(self, inputs, output):

        # all_candidate_labels.shape() == batch_size x max_num_candidates
        all_candidate_label_ids = inputs['candidate_label_ids']
        all_candidate_labels = inputs['candidate_labels']

        print(all_candidate_label_ids)
        print(all_candidate_labels)

        all_candidate_label_ids = all_candidate_label_ids.permute(1, 0, 2)
        all_candidate_labels = all_candidate_labels.permute(1, 0)

        total_loss = 0
        loss_correct_label = loss_fct(output["logits"], all_candidate_label_ids[0].view(-1))

        # compute hinge loss
        for candidate_label_ids, candidate_labels in zip(all_candidate_label_ids[1:], all_candidate_labels[1:]):
            loss_wrong_label = loss_fct(output["logits"], candidate_label_ids.view(-1))
            hinge_loss = 1 + loss_correct_label - loss_wrong_label
            hinge_loss[hinge_loss < 0] = 0
            total_loss += hinge_loss

        loss_fct = CrossEntropyLoss()
        loss_correct_label = loss_fct(output["logits"].view(-1, self.config.vocab_size), correct_targets.view(-1).to(self.model.device))
        loss_wrong_label = loss_fct(output["logits"].view(-1, self.config.vocab_size), wrong_targets.view(-1).to(self.model.device))
        loss = 1 + loss_correct_label - loss_wrong_label
        loss[loss < 0] = 0
        return loss



TASK_HELPERS = {
    "multirc": GeneralTaskHelper,
    "copa": CopaTaskHelper,
    "record": RecordTaskHelper,
    "boolq": GeneralTaskHelper,
    "rte": GeneralTaskHelper,
    "xnli": GeneralTaskHelper,
    "cb": GeneralTaskHelper,
    "wic": GeneralTaskHelper,
    "wsc": WscTaskHelper,
    "paws-x": GeneralTaskHelper,
}