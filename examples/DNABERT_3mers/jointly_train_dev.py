import random

import numpy as np

import torch
import argparse
import os

from time import sleep

import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

import torch.nn as nn

import torch
import wandb
import os.path

from transformers import Trainer, TrainingArguments, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


from dataclasses import dataclass
from typing import List, Dict, Any
import random

import torch
import torch.nn.functional as F

import glob
import os

from  InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA,InstructNAConfig
from  InstructNA_frameworks.utills import kmers_sliding_windows

import torch.distributed as dist


from InstructNA_frameworks.utills import set_seed

import numpy as np

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def read_fastq_sequences_no_N(fastq_path, prefix=None):
    sequences = []
    with open(fastq_path, "r") as f:
        while True:
            header = f.readline()
            seq = f.readline()
            plus = f.readline()
            qual = f.readline()

            if not qual:
                break

            seq = seq.strip().upper()
            if "N" not in seq:
                sequences.append((prefix +"|"+ seq) if prefix else seq)
    return sequences




class SELEX_Dataset(Dataset):
    def __init__(self, seq_dir, DNA_tokenizer, num_workers=8):
        self.DNA_tokenizer = DNA_tokenizer

        all_seqs = []
        with open(seq_dir, 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:
                    all_seqs.append(kmers_sliding_windows(seq))

        self.all_seqs = all_seqs
        self.data_len = len(self.all_seqs)
        
        tokens_ids = DNA_tokenizer.batch_encode_plus(self.all_seqs, return_tensors="pt", padding="longest")["input_ids"]
        self.data=tokens_ids
        

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx].clone().detach().long()}


def mask_tokens(input_ids: torch.Tensor, tokenizer, mlm_probability: float = 0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=labels.device)
    special_tokens_mask = torch.zeros_like(labels).bool()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()


    left_shifted = torch.nn.functional.pad(masked_indices[:, :-1], (1,0), value=False)
    right_shifted = torch.nn.functional.pad(masked_indices[:, 1:], (0,1), value=False)

    masked_indices = masked_indices | left_shifted | right_shifted

    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(low=0, high=tokenizer.vocab_size, size=labels.shape, device=labels.device)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels










def train():
    parser = argparse.ArgumentParser(description="Train InstructNA model with DNABERT_3mers")


    parser.add_argument(
        '--dataset_dir', type=str, default="/home/hanlab/2022_zzm/InstructNA_new/data/CELL_SELEX_2013/SELEX_data/Ar_TCTAAT20NCG_P_5/unique_seqs.csv", required=False,
        help="Path to the dataset file. The file should contain one DNA/RNA sequence per line."
    )

    parser.add_argument(
        '--batchsize', type=int, default=512, required=False,
        help="Number of samples per batch during training."
    )

    parser.add_argument(
        '--train_val_split_ratio', type=float, default=0.9, required=False,
        help="Ratio to split the dataset into training and validation sets. "
            "For example, 0.9 means 90% training, 10% validation."
    )

    parser.add_argument(
        '--train_epoches', type=int, default=10, required=False,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--dataloader_num_worker', type=int, default=8, required=False,
        help="Number of worker processes used by the DataLoader for loading batches."
    )

    parser.add_argument(
        '--dataset_preprocess_num_worker', type=int, default=8, required=False,
        help="Number of worker processes used for preprocessing the dataset before training."
    )

    parser.add_argument(
        '--evaluate_per_step', type=int, default=500, required=False,
        help="Perform evaluation every N training steps."
    )

    parser.add_argument(
        '--random_seeds', type=int, default=42, required=False,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--checkpoint_save_path', type=str, default="/home/hanlab/2022_zzm/InstructNA_new/output/model_save/jointly_test", required=False,
        help="Directory path to save model checkpoints during training."
    )

    parser.add_argument(
        '--max_checkpoint_save_num', type=int, default=3, required=False,
        help="Maximum number of model checkpoints to keep. Older checkpoints will be deleted."
    )

    parser.add_argument(
        '--wandb_exp_name', type=str, default="IN_jointly_train", required=False,
        help="Weights & Biases experiment name for logging (if using W&B)."
    )

    parser.add_argument(
        '--warmup_steps', type=int, default=100, required=False,
        help="Number of warm-up steps for the learning rate scheduler."
    )

    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1, required=False,
        help="Number of gradient accumulation steps before performing an optimizer step."
    )

    parser.add_argument(
        '--lr', type=float, default=1e-4, required=False,
        help="Initial learning rate for the optimizer."
    )

    parser.add_argument(
        '--device', type=str, default="cuda", required=False,
        help="Device to run training on. Options: 'cuda', 'cpu'."
    )


    parser.add_argument(
        '--model_down_dim', type=int, default=8, required=False,
        help="Dimension of the reduced latent space (used for dimensionality reduction in the model)."
    )

    
    args = parser.parse_args()
    max_checkpoint_save_num=args.max_checkpoint_save_num
    dataset_dir = args.dataset_dir
    batchsize = args.batchsize
    train_val_split_ratio = args.train_val_split_ratio
    train_epoches = args.train_epoches
    wandb_exp_name=args.wandb_exp_name
    gradient_accumulation_steps=args.gradient_accumulation_steps
    model_down_dim=args.model_down_dim
    
    lr=args.lr
    dataloader_num_worker=args.dataloader_num_worker



    step_to_evaluate = args.evaluate_per_step
    random_seed=args.random_seeds


    if args.checkpoint_save_path:
        checkpoint_save_path = args.checkpoint_save_path
    else:
        if wandb_exp_name:
            checkpoint_save_path = os.path.join(checkpoint_save_path,wandb_exp_name)

    device=args.device

    random.seed(random_seed)
    set_seed(random_seed)


    if wandb_exp_name:
        wandb.login(key="fdc6faaa57d223d86bb721deb5e502f42eb0ea9e")



    # Loading Evo

    print("Loading LM model...")
    config=InstructNAConfig()
    config.proj_dim=model_down_dim
    model=InstructNA(config)
    tokenizer=model.tokenizer
    print(model)
    

    

    trainable_params = 0
    for param in model.parameters():
        param.requires_grad = False
    tuning_layers=[]
    for name, param in model.named_parameters():
        #full tune
        print(f"{name} layer is training")
        tuning_layers.append(name)
        param.requires_grad = True
        trainable_params += param.numel()
    print(f"Total trainable parameters: {trainable_params / 1e6:.2f} M")


    
    # Loading Dataloader
    DNA_Protein_dataset = SELEX_Dataset(dataset_dir, DNA_tokenizer=tokenizer)
    train_size = int(train_val_split_ratio * len(DNA_Protein_dataset))
    val_size = len(DNA_Protein_dataset) - train_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(DNA_Protein_dataset, [train_size, val_size], generator=generator)

    
    model.to(device)
    training_args = TrainingArguments(
    dataloader_num_workers=dataloader_num_worker,
    output_dir=checkpoint_save_path,
    eval_strategy="steps",
    learning_rate=lr,
    per_device_train_batch_size=batchsize,
    per_device_eval_batch_size=batchsize,
    gradient_accumulation_steps =gradient_accumulation_steps,
    num_train_epochs=train_epoches,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=step_to_evaluate,
    save_steps =step_to_evaluate,
    save_total_limit=max_checkpoint_save_num,
    report_to="wandb",  
    bf16=True,  
    save_safetensors=False,
    logging_first_step=True
    )

    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            记录：
            - total loss
            - reconstruction loss
            - mlm loss（如果存在）
            """

            outputs = model(**inputs)
            
            if isinstance(outputs, dict):
                loss = outputs["loss"]
                loss_rec = outputs.get("rec_loss", None)
                loss_mlm = outputs.get("mlm_loss", None)
            else:
                loss = outputs.loss
                loss_rec = getattr(outputs, "rec_loss", None)
                loss_mlm = getattr(outputs, "mlm_loss", None)

            log_dict = {
                "loss_total": loss.detach().item(),
            }

            if loss_rec is not None:
                log_dict["loss_rec"] = loss_rec.detach().item()

            if loss_mlm is not None:
                log_dict["loss_mlm"] = loss_mlm.detach().item()

            self.log(log_dict)

            return (loss, outputs) if return_outputs else loss
        def prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=None,
                **kwargs,
            ):
            loss, logits, labels = super().prediction_step(
                model,
                inputs,
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
            )

            if isinstance(logits, tuple):
                outputs = logits
            else:
                outputs = None

            if hasattr(model, "last_outputs"):
                outputs = model.last_outputs

            with torch.no_grad():
                outputs = model(**inputs)

            log_dict = {}

            if isinstance(outputs, dict):
                if "rec_loss" in outputs:
                    log_dict["eval_loss_rec"] = outputs["rec_loss"].detach().item()
                if "mlm_loss" in outputs:
                    log_dict["eval_loss_mlm"] = outputs["mlm_loss"].detach().item()

            if log_dict:
                self.log(log_dict)

            return loss, logits, labels
                
        


    def custom_collator(batch):


        x_list = [item["input_ids"] for item in batch]
        x_padded = pad_sequence(x_list, batch_first=True, padding_value=1)  
        input_ids, labels = mask_tokens(x_padded.clone().detach().long(),tokenizer)
        return {    
            "full_input_ids": x_padded, 
            "input_ids": input_ids,
            "labels": labels
        }


    trainer=CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collator,
    
    )

    trainer.train()


if __name__ == '__main__':
    train()
