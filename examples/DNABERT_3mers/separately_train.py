import os
import glob
import copy
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import wandb
import hydra

from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments, set_seed

from InstructNA_frameworks.models.DNABERT_3mers.model import (
    InstructNA,
    InstructNAConfig,
)
from InstructNA_frameworks.utills import (
    kmers_sliding_windows,
    set_seed as instructna_set_seed,
)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def train_once(cfg):

    random.seed(cfg.training.random_seed)
    set_seed(cfg.training.random_seed)
    
    if cfg.wandb.enable:
        wandb.login(key=cfg.wandb.wandb_key)
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    mode = cfg.model.mode
    if mode not in {"train_encoder", "train_decoder"}:
        raise ValueError("model.mode must be train_encoder or train_decoder")

    if mode == "train_encoder":
        if cfg.model.finetune_encoder_weight_path is not None:
            raise ValueError("Training encoder: finetune path must be None")

        config = InstructNAConfig(proj_dim=cfg.model.down_dim)
        model = InstructNA(config)
        model.train_encoder = True
        print("Training encoder ...")

    else:
        if cfg.model.finetune_encoder_weight_path is None:
            raise ValueError("Training decoder requires encoder checkpoint")

        model = InstructNA.from_pretrained(
            cfg.model.finetune_encoder_weight_path
        )
        model.train_decoder = True
        print("Training decoder ...")

    trainable_params = 0
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if mode == "train_encoder" and "encoder" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"{name} layer is training")

        if mode == "train_decoder" and "decoder" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"{name} layer is training")

    print(f"Total trainable parameters: {trainable_params / 1e6:.2f} M")

    dataset = SELEX_Dataset(
        cfg.dataset.dir,
        DNA_tokenizer=model.tokenizer,
    )

    train_size = int(cfg.dataset.train_val_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(cfg.training.random_seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    model.to(cfg.training.device)

    step_to_eval = (train_size // cfg.training.batchsize //  \
                    cfg.training.gradient_accumulation_steps) // \
                    cfg.training.evaluate_num_per_epoch
        
    training_args = TrainingArguments(
        output_dir=cfg.checkpoint.save_path,
        dataloader_num_workers=cfg.dataloader.num_worker,
        per_device_train_batch_size=cfg.training.batchsize,
        per_device_eval_batch_size=cfg.training.batchsize,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.train_epoches,
        learning_rate=cfg.training.lr,
        lr_scheduler_type="cosine",
        eval_steps= step_to_eval,
        logging_steps= 1,
        save_steps=step_to_eval,
        save_total_limit=cfg.checkpoint.max_save_num,
        eval_strategy="steps",
        report_to="wandb" if cfg.wandb.enable else "none",
        save_safetensors=False,
        logging_first_step=True,
    )

    def custom_collator(batch):
        x_list = [item["input_ids"] for item in batch]
        x_padded = pad_sequence(x_list, batch_first=True, padding_value=1)
        input_ids, labels = mask_tokens(
            x_padded.clone().detach().long(), model.tokenizer
        )
        return {
            "full_input_ids": x_padded,
            "input_ids": input_ids,
            "labels": labels,
        }

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs["loss"]

            log_dict = {"loss_total": loss.detach().item()}
            if outputs.get("rec_loss") is not None:
                log_dict["loss_rec"] = outputs["rec_loss"].detach().item()
            if outputs.get("mlm_loss") is not None:
                log_dict["loss_mlm"] = outputs["mlm_loss"].detach().item()

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
                if outputs.get("rec_loss") is not None:
                    log_dict["eval_loss_rec"] = outputs["rec_loss"].detach().item()
                if outputs.get("mlm_loss") is not None:
                    log_dict["eval_loss_mlm"] = outputs["mlm_loss"].detach().item()

            if log_dict:
                self.log(log_dict)

            return loss, logits, labels
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.tokenizer,
        data_collator=custom_collator,
    )

    trainer.train()
    trainer.save_model(cfg.checkpoint.save_path)
    wandb.finish()
    return trainer
    



@hydra.main(config_path="conf", config_name="separately_train", version_base="1.3")
def main(cfg):

    if cfg.run.mode == "separate":

        print("One-click separate training: encoder → decoder")

        # ---------- encoder ----------
        encoder_cfg = copy.deepcopy(cfg)
        encoder_cfg.model.mode = "train_encoder"
        encoder_cfg.model.finetune_encoder_weight_path = None
        encoder_cfg.checkpoint.save_path = os.path.join(
            cfg.checkpoint.save_path, "encoder"
        )

        train_once(encoder_cfg)

        # ---------- decoder ----------
        decoder_cfg = copy.deepcopy(cfg)
        decoder_cfg.model.mode = "train_decoder"
        decoder_cfg.model.finetune_encoder_weight_path = (
            encoder_cfg.checkpoint.save_path
        )
        decoder_cfg.checkpoint.save_path = os.path.join(
            cfg.checkpoint.save_path, "decoder"
        )

        final_trainer = train_once(decoder_cfg)
        final_trainer.save_model(os.path.join(cfg.checkpoint.save_path, "final_model"))

        print("Finished encoder + decoder training")

    else:
        train_once(cfg)



if __name__ == "__main__":
    main()

