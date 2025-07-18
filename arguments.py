from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import *


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # NOTE 没用到
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    # NOTE 没用到
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    # NOTE 没用到
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    zero_tuning:Optional[str] = field(
        default=None, 
        metadata={"help": "whether to using test to zero_tuning"}
    )
    ensemble_lang:Optional[str] = field(
        default=None, 
        metadata={"help": "language use for ensemble"}
    )
    generate_train:Optional[str] = field(
        default=None, 
        metadata={"help": "using SuperGen data to train"}
    )
    all_lang_prompt:Optional[str] = field(
        default=None, 
        metadata={"help": "using SuperGen data to train"}
    )
    self_training:Optional[str] = field(
        default=None, 
        metadata={"help": "whether to use unlabel data to self training"}
    )
    parallel_training:Optional[str] = field(
        default=None, 
        metadata={"help": "whether to use few shot parallel data to train"}
    )
    ensemble_lang_num:Optional[int] = field(
        default=0, 
        metadata={"help": "get top N language to ensemble learning"}
    )
    ensemble_lang_connect: bool = field(
        default=False, 
        metadata={"help": "whether only to bulid connection between week and strong languages"}
    )
    ensemble_lang_pair: Optional[int] = field(
        default=0, 
        metadata={"help": "get top N and last N language pair to ensemble learning"}
    )
    ensemble_weak_number: Optional[int] = field(
        default=0, 
        metadata={"help": "get N language pair as weak language to ensemble learning"}
    )
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # NOTE 没用到
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # NOTE 没用到
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # NOTE 没用到
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # NOTE 没用到
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # NOTE 没用到
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    pre_seq_len: int = field(
        default=6,
        metadata={
            "help": "The length of prompt"
        }
    ) # 可能会引起误会，datasets内也定义了pre_seq_len
    task_type: Optional[str] = field(
        default="language_modeling",
        metadata={
            "help": "Design which head to use."
        }
    )
    eval_type: Optional[str] = field(
        default="eval",
        metadata={
            "help": "Design which head to use."
        }
    )
    prompt_type: Optional[str] = field(
        default="soft",
        metadata={
            "help": "Use hard or soft prompt"
        }
    )
    template_id: Optional[str] = field(
        default="template_0",
        metadata={
            "help": "The specific soft prompt template to use"
        }
    )
    verbalizer_id: Optional[str] = field(
        default="verbalizer_0",
        metadata={
            "help": "The specific verbalizer to use"
        }
    )
    prompt_operation: Optional[str] = field(
        default="mean",
        metadata={
            "help": "Will use max, sum, mean, attention or cross-attention soft prompt tuning during training"
        }
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    )
    prefix_hidden_size: int = field(
        default=768,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    ) # 可能会引起误会
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    num_attention_layers: int = field(
        default=1,
        metadata={
            "help": ""
        }
    )
    num_attention_heads: int = field(
        default=8,
        metadata={
            "help": ""
        }
    )
    whether_PositionalEncoding: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    whether_PositionalWiseFeedForward: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    fix_deberta: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    data_augmentation: Optional[str] = field(
        default="none",
        metadata={
            "help": "rdrop, AT, mixup, manifold_mixup"
        }
    )
    device: Optional[str] = field(
        default="0",
        metadata={
            "help": "Gpu device num"
        }
    )

@dataclass
class QuestionAnwseringArguments:
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )

def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, QuestionAnwseringArguments))

    args = parser.parse_args_into_dataclasses()

    return args