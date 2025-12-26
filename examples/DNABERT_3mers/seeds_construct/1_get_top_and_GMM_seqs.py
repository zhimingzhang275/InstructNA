import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from InstructNA_frameworks.utills import (
    fastq_to_csv_with_counts,
    set_seed,
    get_cluster_center_seqs,
    read_fastq_dedup_no_N
)
from InstructNA_frameworks import InstructNA

@hydra.main(version_base=None, config_path="../conf/seeds_construct/", config_name="top_and_gmm_seqs")
def main(cfg: DictConfig):

    print("===== CONFIG =====")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.base.seed)
    device = cfg.base.device

    all_seqs=[]
    if cfg.top.enable:
        if cfg.base.data.fastq_dir is None:
            raise ValueError("fastq file must be provided")
        os.makedirs(cfg.base.output_dir, exist_ok=True)
        output_file=os.path.join(cfg.base.output_dir,"top_sequences.csv")
        seq_counter, _ = fastq_to_csv_with_counts(
            cfg.base.data.fastq_dir,
            output_file
        )

        topk = seq_counter[: cfg.top.top_k]

        with open(output_file, "w") as f:
            for seq, count in topk:
                all_seqs.append(seq)
                f.write(f"{seq}\n")

        print(f"[FASTQ] Top {cfg.top.top_k} sequences saved to {output_file}")


    if cfg.gmm.enable:

        print("Loading model...")
        model = InstructNA.from_pretrained(cfg.base.model.InstructNA_model_path)
        tokenizer = model.tokenizer
        
        model.to(device)

        all_SELEX_unique_seq=read_fastq_dedup_no_N(cfg.base.data.fastq_dir)
        generated_seqs, _, _, _ = get_cluster_center_seqs(
            all_SELEX_unique_seq,
            model,
            tokenizer,
            tSNE=cfg.gmm.tsne_visual,
            model_down_dim=model.config.proj_dim
        )

        output_path = os.path.join(cfg.base.output_dir, "GMM_center_seqs.csv")
        with open(output_path, "w") as f:
            for seq in generated_seqs:
                all_seqs.append(seq)
                print(f"[GMM] Generated sequence: {seq}")
                f.write(f"{seq}\n")

        print(f"[GMM] Generated sequences saved to {output_path}")

        with open(os.path.join(cfg.base.output_dir, "all_top_and_GMM_seqs.csv"), "w") as f:
            for seq in all_seqs:
                f.write(f"{seq}\n")
        print(f"[GMM] All top and GMM sequences saved to {os.path.join(cfg.base.output_dir, 'all_top_and_GMM_seqs.csv')}")
if __name__ == "__main__":
    main()
