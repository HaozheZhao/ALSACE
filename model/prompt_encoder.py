import math
import torch
import torch.nn as nn
from model.transformers import Encoder, crossEncoder

class EmbeddingEncoder():
    def __init__(self, config, model_args, prompt_encoder, model):
        self.config = config
        self.model_args = model_args
        self.prompt_encoder = prompt_encoder
        self.model = model

    def id2embedding(self, input_ids, sentences_ids, label_token_id_list):

        # construct query ids
        batch_size = input_ids.shape[0]

        if label_token_id_list:
            if isinstance(label_token_id_list[0][0].item(), list):
                num_mask_token = max(len(length.item()) for length in label_token_id_list[0])
            elif isinstance(label_token_id_list[0][0].item(), int):
                num_mask_token = 1

        if self.model_args.prompt_operation in ["attention", "cross-attention"]:
            attention_mask_sentences_ids = sentences_ids != self.model_args.tokenizer.pad_token_id # batch_size * seq_length
            prompts = self.prompt_encoder(sentences_ids, attention_mask_sentences_ids) # batch_size * pre_seq_length * hidden_size
        elif self.model_args.prompt_operation in ["max", "mean", "sum"]:
            prompts = self.prompt_encoder(sentences_ids) # batch_size * max_seq_len
            prompts = torch.ones(batch_size, self.model_args.pre_seq_len, self.config.hidden_size).to(self.bert.device)
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    prompts[batch_id, seq_id, :] = self.prompt_encoder(sentences_ids[batch_id][sentences_ids[batch_id] != self.model_args.tokenizer.pad_token_id])[:] # 这一步已经把pad_token_id排除了。
        elif self.model_args.prompt_operation in ["none"]:
            prompts = torch.ones(batch_size, self.model_args.pre_seq_len, self.config.hidden_size).to(self.bert.device)
            prompts_replace = self.prompt_encoder()
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    prompts[batch_id, seq_id, :] = prompts_replace if self.model_args.pre_seq_len == 1 else prompts_replace[seq_id, :]
        else:
            raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))

        attention_mask = input_ids != self.model_args.tokenizer.pad_token_id # batch_size * seq_length

        # get embedded input
        # inputs_embeds = self.embed_input(input_ids, prompts, self.args) # batch_size * max_seq_length * embedding_dim
        input_for_embedding = input_ids.clone()
        input_for_embedding[(input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0])] = self.model_args.tokenizer.unk_token_id # 转化[PROMPT]_id为[UNK]_id
        inputs_embeds = self.bert.get_input_embeddings()(input_for_embedding) # 转化token_id [batch_size, seq_len]为 embedding [batch_size, seq_len, embedding_dim]

        if self.model_args.prompt_type == "none":
            pass
        else:
            # blocked_indices = (input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0]).nonzero().reshape((batch_size, self.model_args.pre_seq_len, -1))[:, :, 1]  # 把[PROMPT]标记为1，反转，[PROMPT]标记为0，返回的是[PROMPT]的索引
            blocked_indices = torch.nonzero(input_ids == self.model_args.tokenizer.additional_special_tokens_ids[0]).reshape((batch_size, self.model_args.pre_seq_len, -1))[:, :, 1]  # 把[PROMPT]标记为1，反转，[PROMPT]标记为0，返回的是[PROMPT]的索引
            for batch_id in range(batch_size):
                for seq_id in range(self.model_args.pre_seq_len):
                    inputs_embeds[batch_id, blocked_indices[batch_id, seq_id], :] = prompts[batch_id, seq_id, :]

        return inputs_embeds

    def logits2pred(self, input_ids, prediction_scores, label_token_id_list):
        # 这个地方的logit要取到只有标签
        # 要改成soft label也是在这里改

        batch_size = input_ids.shape[0]

        # label_token_id
        # label_ids_all = torch.LongTensor([[self.model_args.tokenizer.encode(l)[1:-1] for l in self.labels] for _ in range(batch_size)]).squeeze(2).to(self.bert.device) # batch_size * num_labels

        # label_mask = (input_ids == self.model_args.tokenizer.mask_token_id).nonzero().reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
        label_mask = torch.nonzero(input_ids == self.model_args.tokenizer.mask_token_id).reshape(batch_size, num_mask_token, -1)[:, :, -1].to(self.bert.device) # batch_size * num_mask
        
        vocab_size = prediction_scores.shape[-1] # vocab_size
        index = label_mask.unsqueeze(2).repeat(1, 1, vocab_size).long() # batch_size * 1 * vocab_size
        y_pred = torch.gather(prediction_scores, index=index, dim=1).squeeze(1) # batch_size * vocab_size
        y_pred = torch.gather(y_pred, index=label_token_id_list, dim=1) # batch_size  * num_labels
        # y_rank = y_pred.argmax(axis=1) # batch_size
        # 这个地方不用argmax，留下带有两个选项的logit值的tensor就好。因为compute_metric自己会argmax。
        if self.model_args.multiple_choice:
            y_pred = y_pred[:, 0].unsqueeze(1).reshape(batch_size // 2, 2).repeat(1, 2).reshape(batch_size, 2)
        # print(y_pred.shape)

        return y_pred



class PromptEncoder(torch.nn.Module):
    def __init__(self, config, model_args, model):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model = model

        # self.cloze_length = template
        # self.cloze_mask = [
        #     [1] * self.cloze_length[0]  # first cloze
        #     + [1] * self.cloze_length[1]  # second cloze
        #     + [1] * self.cloze_length[2]
        # ]
        self.cloze_mask = [[1] * self.model_args.pre_seq_len] #[[1, 1, 1, 1]]
        self.cloze_mask = torch.LongTensor(self.cloze_mask) #[[True, True, True]]

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))) # [[0, 1, 2, 3, 4]]
        # embedding
        self.nomal_embeddings = torch.nn.Embedding(self.model_args.pre_seq_len, self.config.hidden_size)
        # embedding
        self.specific_embeddings = self.model.get_input_embeddings()
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.config.hidden_size,
                                       hidden_size=self.config.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.model_args.hidden_dropout_prob,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                      nn.ReLU())
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )

        if self.model_args.prompt_operation == "attention":
            self.global_embeded = nn.Parameter(torch.Tensor(self.model_args.pre_seq_len, self.config.hidden_size)).to(self.model.device) # pre_seq_len, hidden_size
            nn.init.kaiming_uniform_(self.global_embeded, a=math.sqrt(5))

            self.transformers = Encoder(
                max_seq_len=self.config.max_position_embeddings,
                num_layers=self.model_args.num_attention_layers,
                model_dim=self.config.hidden_size,
                num_heads=self.model_args.num_attention_heads,
                ffn_dim=self.config.intermediate_size,
                dropout=self.model_args.hidden_dropout_prob,
                whether_PositionalEncoding=self.model_args.whether_PositionalEncoding,
                whether_PositionalWiseFeedForward=self.model_args.whether_PositionalWiseFeedForward
            )
        elif self.model_args.prompt_operation == "cross-attention":
            self.global_embeded = nn.Parameter(torch.Tensor(self.model_args.pre_seq_len // 2, self.config.hidden_size)).to(self.model.device) # pre_seq_len // 2, hidden_size
            nn.init.kaiming_uniform_(self.global_embeded, a=math.sqrt(5))
            self.local_embeded = torch.nn.Embedding(self.model_args.pre_seq_len // 2, self.config.hidden_size)
            self.local_seq_indices = torch.LongTensor(list(range(self.model_args.pre_seq_len // 2))) # [[0, 1, 2, 3, 4]]
            self.transformers = crossEncoder(
                max_seq_len=self.config.max_position_embeddings,
                num_layers=self.model_args.num_attention_layers,
                model_dim=self.config.hidden_size,
                num_heads=self.model_args.num_attention_heads,
                ffn_dim=self.config.intermediate_size,
                dropout=self.model_args.hidden_dropout_prob,
                whether_PositionalEncoding=self.model_args.whether_PositionalEncoding,
                whether_PositionalWiseFeedForward=self.model_args.whether_PositionalWiseFeedForward
            )
        print("init prompt encoder...")

    def forward(self, sentences_encoded=None, attention_mask=None):
        if sentences_encoded != None:
            batch_size = sentences_encoded.size(0)
        if self.model_args.prompt_operation in ["attention"]:
            global_embeded = self.global_embeded.unsqueeze(0).repeat(batch_size, 1, 1)# batch_size * pre_seq_len, hidden_size
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # batch_size * seq_len * hidden_size
            cat_embeded = torch.cat((global_embeded, sentences_embeded), dim=1) # batch_size * total_seq_len * hidden_size
            cat_attention = torch.cat((torch.ones(batch_size, self.model_args.pre_seq_len).bool().to(self.model.device), attention_mask), dim=1) # batch_size * total_seq_len * hidden_size
            prompts, _ = self.transformers(cat_embeded, cat_attention) # batch_size * total_seq_len * hidden_size
            if self.model_args.task_type == "sequence_classification":
                prompts = self.trans(prompts)
            result = prompts[:, :self.model_args.pre_seq_len, :] # batch_size * pre_seq_len * hidden_size
            return result
        elif self.model_args.prompt_operation in ["cross-attention"]:
            global_embeded = self.global_embeded.unsqueeze(0).repeat(batch_size, 1, 1)# batch_size, pre_seq_len // 2, hidden_size
            local_embeded = self.local_embeded(self.local_seq_indices.to(self.model.device)).detach().unsqueeze(0).repeat(batch_size, 1, 1) # batch_size,  pre_seq_len // 2, hidden_size
            cat_embeded = torch.cat((global_embeded, local_embeded), dim=1) # batch_size * total_seq_len * hidden_size
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # batch_size * seq_len * hidden_size
            prompts, _ = self.transformers(cat_embeded, sentences_embeded, attention_mask) # batch_size * total_seq_len * hidden_size
            if self.model_args.task_type == "sequence_classification":
                prompts = self.trans(prompts)
            return prompts
        elif self.model_args.prompt_operation in ["mean", "sum", "max"]:
            sentences_embeded = self.specific_embeddings(sentences_encoded).detach() # seq_len * hidden_size
            output_embeded = self.mlp_head(sentences_embeded).squeeze() # seq_len * hidden_size
            if self.model_args.prompt_operation == "mean":
                result = torch.mean(output_embeded, 0) # hidden_size
            elif self.model_args.prompt_operation == "sum":
                result = torch.sum(output_embeded, 0) # hidden_size
            elif self.model_args.prompt_operation == "max":
                result = torch.max(output_embeded, 0).values # hidden_size
            elif self.model_args.prompt_operation == "attention":
                pass
            else:
                raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))
            return result
        elif self.model_args.prompt_operation in ["none"]:
            input_embeds = self.nomal_embeddings(self.seq_indices.to(self.model.device)).detach().unsqueeze(0) # seq_len * hidden_size
            # LSTM_embeds = self.lstm_head(input_embeds) # seq_len * hidden_size
            output_embeds = self.mlp_head(input_embeds).squeeze() # seq_len * hidden_size
            return output_embeds
        else:
            raise NotImplementedError("The prompt_operation for {} has not been defined.".format(self.model_args.prompt_operation))

