
import argparse
import logging
import os
import random
import json

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)

from t0.data_collator import DataCollatorForMultipleChoice
from t0.model import ModelBase


def evaluation(accelerator, tokenizer, model, args, eval_dataset, logger, task_name):
    # Log a few random samples from the eval set:
    prediction_gather = []
    target_gather = []
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # Use the device given by the `accelerator` object.
    if not args.parallelize:
        model.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(eval_dataloader)


    # Metrics
    metric = load_metric("accuracy")

    # Eval!
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            predictions = model(batch)
            
            if task_name == "negation":
                prediction_gather.append(torch.Tensor([1-x for x in predictions]).type(torch.int))
            else:
                prediction_gather.append(predictions)
            target_gather.append(batch["targets"])

        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["targets"]),
        )

        progress_bar.update(1)

    eval_metric = metric.compute()
    accelerator.print(f"Result: {eval_metric}")

    results = {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "template_name": args.template_name,
        "task_name" : task_name,
        "evaluation": eval_metric
    }
    if accelerator.is_main_process:
        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, f"{task_name}.json"), "w") as f:
                json.dump(results, f, indent=4)
    return prediction_gather, target_gather

def evaluation_agreement(accelerator, tokenizer, model, args, eval_dataset_or_list, target_dataset_or_list, logger, task_name):
    if type(eval_dataset_or_list)!= list:
        eval_list, _ = evaluation(accelerator, tokenizer, model, args, eval_dataset_or_list, logger, task_name)
    else:
        eval_list = eval_dataset_or_list
    if type(target_dataset_or_list)!= list:
        target_list, _ = evaluation(accelerator, tokenizer, model, args, target_dataset_or_list, logger, task_name) 
    else:
        target_list = target_dataset_or_list   

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    # Metrics
    metric = load_metric("accuracy")

    # Eval!
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation agreement *****")
    logger.info(f"  Num examples = {len(eval_list)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_list)), disable=not accelerator.is_local_main_process)

    model.eval()
    for i in range(len(eval_list)):

        metric.add_batch(
            predictions=accelerator.gather(eval_list[i]),
            references=accelerator.gather(target_list[i]),
        )

        progress_bar.update(1)

    eval_metric = metric.compute()
    accelerator.print(f"Result: {eval_metric}")

    results = {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "template_name": args.template_name,
        "task_name": task_name,
        "evaluation": eval_metric
    }
    if accelerator.is_main_process:
        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, f"{task_name}.json"), "w") as f:
                json.dump(results, f, indent=4)