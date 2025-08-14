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

from  InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
from  InstructNA_frameworks.utills import kmers_sliding_windows,load_latest_checkpoint

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
        return {"x": self.data[idx].clone().detach().long()}


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



def custom_collator(batch):
    """
    Process variable-length DNA sequences by padding them so that all "x" sequences 
    in the batch have the same length.

    Args:
        batch (List[Dict]): Each element is a sample returned by the Dataset, 
            e.g., {"x": torch.tensor(DNA)}

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the batch data
            - "x": (batch_size, max_seq_len) tensor after padding
            - "attention_mask": (batch_size, max_seq_len) tensor where 1 marks valid positions 
              and 0 marks padding positions (not included here but can be added if needed)
    """

    x_list = [item["x"] for item in batch]
    x_padded = pad_sequence(x_list, batch_first=True, padding_value=1)  

    return {
        "x": x_padded, 
    }



def save_partial_model(model, itearation,layers_to_save=["8","16","24"], save_dir_path="./output/best_checkpoint/"):
    state_dict = {k: v for k, v in model.state_dict().items() if any(layer in k for layer in layers_to_save)}
    
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    final_saved_path=os.path.join(save_dir_path, f"checkpoint_{itearation}.pth")
    torch.save(state_dict, final_saved_path)




def train():
    parser = argparse.ArgumentParser(description="Train InstructNA model with DNABERT_3mers")


    parser.add_argument(
        '--dataset_dir', type=str, default=None, required=False,
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
        '--train_epoches', type=int, default=50, required=False,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--dataloader_num_worker', type=int, default=64, required=False,
        help="Number of worker processes used by the DataLoader for loading batches."
    )

    parser.add_argument(
        '--dataset_preprocess_num_worker', type=int, default=8, required=False,
        help="Number of worker processes used for preprocessing the dataset before training."
    )

    parser.add_argument(
        '--evaluate_per_step', type=int, default=10, required=False,
        help="Perform evaluation every N training steps."
    )

    parser.add_argument(
        '--random_seeds', type=int, default=42, required=False,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--checkpoint_save_path', type=str, default=None, required=False,
        help="Directory path to save model checkpoints during training."
    )

    parser.add_argument(
        '--max_checkpoint_save_num', type=int, default=3, required=False,
        help="Maximum number of model checkpoints to keep. Older checkpoints will be deleted."
    )

    parser.add_argument(
        '--wandb_exp_name', type=str, default=None, required=False,
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
        '--train_encoder_or_decoder', type=str, default="train_decoder", required=False,
        help="Which part of the model to train. Options: 'train_encoder', 'train_decoder'."
    )

    parser.add_argument(
        '--fintune_encoder_weight_path', type=str, default=None, required=False,
        help="Path to the fine-tuned encoder weights if training the decoder only."
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
    train_encoder_or_decoder=args.train_encoder_or_decoder
    dataloader_num_worker=args.dataloader_num_worker
    fintune_encoder_weight_path=args.fintune_encoder_weight_path


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
    # config = AutoConfig.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    
    
    print(model.config)
    
    lm_embeds_dim = model.config.hidden_size
    lm_vocab_size = model.config.vocab_size
    model=InstructNA(model,lm_embeds_dim,lm_vocab_size,proj_dim=model_down_dim)
    if train_encoder_or_decoder=="train_encoder":
        model=model.encoder
    print("Model loaded!")

    
    
    if fintune_encoder_weight_path and train_encoder_or_decoder=="train_decoder":
        print("Loading encoder fintuned weight!")
        if is_main_process():
            finetuned_encoder, _ = load_latest_checkpoint(model.encoder,fintune_encoder_weight_path)
        model.encoder=finetuned_encoder

    

    trainable_params = 0
    for param in model.parameters():
        param.requires_grad = False
    tuning_layers=[]
    if train_encoder_or_decoder=="train_encoder":
        for name, param in model.named_parameters():
            #full tune
            print(f"{name} layer is training")
            tuning_layers.append(name)
            param.requires_grad = True
            trainable_params += param.numel()

        print(f"Total trainable parameters: {trainable_params / 1e6:.2f} M")
    else:
        for name, param in model.named_parameters():
            # finetune attention layer
            if "decoder" in name:
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
    report_to=wandb_exp_name,  
    bf16=True,  
    save_safetensors=False,
    logging_first_step=True
    )

    

        
    def compute_metrics(eval_pred):
     
        logits, labels = eval_pred.predictions, eval_pred.label_ids

        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)

        # Flatten
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.view(-1)
        valid_mask = shift_labels != -100
        valid_logits = shift_logits[valid_mask]
        valid_labels = shift_labels[valid_mask]
        ce_loss = torch.nn.functional.cross_entropy(
            valid_logits,
            valid_labels,

        ).item()

  
        perplexity = np.exp(ce_loss)
        predictions = valid_logits.argmax(dim=-1)
        correct = (predictions == valid_labels).sum().item()
        total = valid_labels.size(0)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "eval_loss": ce_loss,
            "perplexity": perplexity,
            "accuracy": accuracy
        }
        
        
    class CustomTrainer(Trainer):
        def __init__(self, save_layers=None, **kwargs):
            super().__init__(**kwargs)
            self.save_layers=save_layers
        def compute_loss(self, model, inputs, return_outputs=False,**args):
            dna =inputs["x"]
            
            if train_encoder_or_decoder=="train_encoder":
                input_ids,labels=mask_tokens(dna,tokenizer)
                output = model(input_ids,predict_token=True)  
            else:
                input_ids,labels=dna,dna
                output = model(input_ids)  

            loss = nn.functional.cross_entropy(
                output.reshape(-1, output.shape[-1]),
                labels.long().reshape(-1),
                ignore_index=-100
            )

            if return_outputs:
                return loss, output
            return loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

            model.eval()
            with torch.no_grad():
                dna =inputs["x"]
                if train_encoder_or_decoder=="train_encoder":
                    input_ids,labels=mask_tokens(dna,tokenizer)
                    output = model(input_ids,predict_token=True)  
                else:
                    input_ids,labels=dna,dna
                    output = model(input_ids)  
                logits = output
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.long().reshape(-1),
                    ignore_index=-100
                )
            if prediction_loss_only:
                return (loss, None, None)
            return (loss, logits, labels)
        
        def save_model(self, output_dir: str = None, _internal_call: bool = False):
            """
            Save only the parameters with require_grad=True (i.e., trainable layers) 
            while keeping all other functionalities unchanged.
            The rest of the model will follow the Trainer's default saving logic.
            """
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the complete model state_dict
            state_dict = self.model.state_dict()
            
            # Filter the state_dict to keep only parameters in self.save_layers
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.save_layers}
            
            # Specify the save file name (using .bin suffix here)
            output_model_file = os.path.join(output_dir, "pytorch_model_trainable.bin")
            
            # Save the filtered state_dict
            torch.save(filtered_state_dict, output_model_file)
            
            # Log the save path
            self.log({"filtered_model_saved": output_model_file})



    trainer=CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collator,
        save_layers=tuning_layers,
        compute_metrics=compute_metrics,
    
    )

    trainer.train()


if __name__ == '__main__':
    train()
