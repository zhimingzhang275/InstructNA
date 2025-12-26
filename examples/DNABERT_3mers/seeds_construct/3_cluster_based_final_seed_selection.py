import os
import torch
import argparse
from InstructNA_frameworks.utills import set_seed,select_max_activity_by_cluster,cluster_based_seeds_selection
from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
import hydra
from omegaconf import DictConfig, OmegaConf
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(
    version_base=None,
    config_path="../conf/seeds_construct",
    config_name="cluster_based_final_seeds_selection"
)
def main(cfg: DictConfig):

    print("===== CONFIG =====")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.base.seed)

    if cfg.base.model.InstructNA_model_path is None:
        raise ValueError("model.instructna_model_path must be provided")

    print("Loading InstructNA model...")
    model = InstructNA.from_pretrained(cfg.base.model.InstructNA_model_path)
    tokenizer = model.tokenizer

    device = cfg.base.device
    model.to(device)
    if cfg.base.data.seq_acts is None:
        raise ValueError("data.seq_acts must be provided")

    cluster_based_seeds_selection(
        cfg.base.data.seq_acts,
        model.encoder,
        tokenizer,
        output_dir=cfg.base.output_dir,
        k=cfg.selection_k,
    )

    print(f"Final seeds saved to {cfg.base.output_dir}")

if __name__ == "__main__":
    main()

