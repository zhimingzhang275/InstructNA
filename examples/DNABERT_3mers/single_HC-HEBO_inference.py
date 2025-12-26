import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(version_base=None, config_path="conf", config_name="single_hc_hebo_inference")
def main(cfg: DictConfig):

    from InstructNA_frameworks.utills import (
        set_seed,
        DNABERT_mask_seq_generate_nolinkers,
    )
    from InstructNA_frameworks.BO_utill import run_one_grouped_BO
    from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
    from InstructNA_frameworks.batch_tokenize_func import (
        DNABERT_3mer_tokenize_batch,
    )

    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed.seed_num)

    print("Loading InstructNA model...")
    model = InstructNA.from_pretrained(cfg.paths.InstructNA_model_path)
    tokenizer = model.tokenizer
    print(model.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    SELEX_seqs = []
    if cfg.paths.SELEX_path is not None:
        with open(cfg.paths.SELEX_path, "r") as f:
            SELEX_seqs = [line.strip() for line in f]


    os.makedirs(cfg.paths.BO_output_dir, exist_ok=True)

    all_seq_Kd = run_one_grouped_BO(
        tokenizer=tokenizer,
        InstructNA_model=model,
        device=device,
        Kd_seq_path_or_dict=cfg.paths.seq_act_path,
        output_dir=cfg.paths.BO_output_dir,
        bo_batchsize=cfg.hc_hebo.batchsize,
        model_down_dim=model.config.proj_dim,
        search_r=cfg.hc_hebo.search_r,
        SELEX_seqs=SELEX_seqs,
        max_object=cfg.hc_hebo.maximize,
        tokenize_function=DNABERT_3mer_tokenize_batch,
        decode_func=DNABERT_mask_seq_generate_nolinkers,
        f_linker=cfg.primer.f_primer,
        r_linker=cfg.primer.r_primer,
        use_check_conditions=cfg.hc_hebo.use_filter,
    )

    return all_seq_Kd


if __name__ == "__main__":
    main()
