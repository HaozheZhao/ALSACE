import logging
import os
import sys
import numpy as np
from typing import Dict

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *


logger = logging.getLogger(__name__)


def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None

    ## 是否载入之前的ckpt
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()



def evaluate(trainer,data_args):
    logger.info("*** Evaluate ***")

    if data_args.task_name =='xglue':
        for test in trainer.all_test_dataset:
            metrics,logit,labels = trainer.evaluate(eval_dataset = trainer.all_test_dataset[test])
            lang_type = test.split('.')[1] if "." in test else test
            trainer.log_metrics(f"{lang_type}_test", metrics)
            trainer.save_metrics(f"{lang_type}_test", metrics)
            trainer.save_logit(f"{lang_type}_test", logit,"logit")
            trainer.save_logit(f"{lang_type}_test", labels,"labels")
    elif data_args.task_name =='xtreme':
        for test in trainer.all_test_dataset:
            metrics,logit,labels = trainer.evaluate(eval_dataset = trainer.all_test_dataset[test])
            lang_type = test
            trainer.log_metrics(f"{lang_type}_test", metrics)
            trainer.save_metrics(f"{lang_type}_test", metrics)
            trainer.save_logit(f"{lang_type}_test", logit,"logit")
            trainer.save_logit(f"{lang_type}_test", labels,"labels")



    

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):

        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(
                d, metric_key_prefix="predict"
            )
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":

    
    args = get_args() 

    model_args, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "xglue":
        assert data_args.dataset_name.lower() in XGLUE_DATASETS
        from tasks.xglue.get_trainer import get_trainer
    elif data_args.task_name.lower() == "xtreme":
        assert data_args.dataset_name.lower() in XTREME_DATASETS
        from tasks.xtreme.get_trainer import get_trainer
    else:
        raise NotImplementedError(
            "Task {} is not implemented. Please choose a task from: {}".format(
                data_args.task_name, ", ".join(TASKS)
            )
        )

    set_seed(training_args.seed)

    trainer, predict_dataset = get_trainer(args)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)

    if training_args.do_eval:
        evaluate(trainer,data_args)

    if training_args.do_predict:
        predict(trainer, predict_dataset)

