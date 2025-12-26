
import argparse
import os
import torch
from InstructNA_frameworks.utills import set_seed,kmers_sliding_windows
from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
from transformers import AutoModel, AutoTokenizer
from InstructNA_frameworks.utills import kmers_sliding_windows,select_max_activity_by_cluster\
    ,encode_sequences_to_embeddings,find_nearest_neighbor,fastq_to_csv_with_counts
import hydra
from omegaconf import DictConfig, OmegaConf
from InstructNA_frameworks.utills import cluster_based_seeds_selection

def get_near_seq(seq_fre_data_dir,seqs_act_dir,encode_model=None,tokenizer=None,output_dir=None):
    device = next(encode_model.parameters()).device
    seqs_act = {}

    with open(seqs_act_dir, "r") as f:
        first_line = f.readline().strip()
        try:
            seqs, act = first_line.split(",")
            float(act)  
            seqs_act[seqs] = float(act)
        except ValueError:
            pass

        for line in f:
            seqs, act = line.strip().split(",")
            seqs_act[seqs] = float(act)
    seeds=set(seqs_act.keys())
    s3_center_seq,s3_center_act=select_max_activity_by_cluster(seqs_act, encode_model, tokenizer)
    
    all_seq_fre_dict={}
    with open(seq_fre_data_dir,"r")as f:
        lines=f.readlines()
        for line in lines[1:]:  # Skip header
            seq,act=line.strip().split(",")
            all_seq_fre_dict.update({seq:float(act)})
    
    cat_seq_embeds, sequence_dict, all_embeds_seq_dict=encode_sequences_to_embeddings(encode_model,tokenizer,list(all_seq_fre_dict.keys()))
    mers_sep_seq=[kmers_sliding_windows(seq) for seq in s3_center_seq]
    input_ids = tokenizer.batch_encode_plus(mers_sep_seq, add_special_tokens=True, max_length=512,truncation=True)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device)

    with torch.no_grad():
        output = encode_model(input_ids)["latent"]
        s3_center_seq_embeddings = output.view(output.shape[0], -1)  # [batch_size, embedding_dim]

    s3=[]
    for embed in s3_center_seq_embeddings.cpu().tolist():
        embed=torch.tensor(embed).unsqueeze(0).to(device)
        found_near_tensor = find_nearest_neighbor(embed, cat_seq_embeds,k=11)
        q_embedding = found_near_tensor.tolist()
        embed_near_seq_list = []

        for idx,embed in enumerate(q_embedding):
            find_seq=all_embeds_seq_dict[tuple(embed)]
            if idx>0 and idx<11:
                print(f"Found near sequence: {find_seq}")
            embed_near_seq_list.append([find_seq,all_seq_fre_dict[find_seq]])
        chosen_seq_list=sorted(embed_near_seq_list, key=lambda x: x[1], reverse=True)
        idx=0
        while (chosen_seq_list[idx][0] in seeds or chosen_seq_list[idx][0] in s3) :
            idx+=1

        s3.append(chosen_seq_list[idx][0])
        
    seeds.update(s3)
    
    
    for seq in s3:
        with open(output_dir, "a") as f:
            f.write(f"{seq}\n")

    return s3




@hydra.main(
    version_base=None,
    config_path="../conf/seeds_construct",
    config_name="get_near_seqs"
)
def main(cfg: DictConfig):

    print("===== CONFIG =====")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.base.seed)
    device = cfg.base.device 

    print("Loading InstructNA model...")

    if cfg.base.model.InstructNA_model_path is None:
        raise ValueError("model.InstructNA_model_path must be provided")

    model = InstructNA.from_pretrained(cfg.base.model.InstructNA_model_path)
    tokenizer = model.tokenizer

    print(model.config)
    model.to(device)

    if cfg.base.data.seq_acts is None:
        raise ValueError("data.seq_acts must be provided")
    s1_s2_selected_path = os.path.join(cfg.base.output_dir, "s1_s2_selected.csv")
    cluster_based_seeds_selection(cfg.base.data.seq_acts, model.encoder, tokenizer, s1_s2_selected_path,k=10)
    s3_seqs_path=os.path.join(cfg.base.output_dir, "near_seqs_in_SELEX.csv")
    _,seq_fre_dir=fastq_to_csv_with_counts(cfg.base.data.fastq_dir, os.path.join(cfg.base.output_dir, "fastq_counts.csv"))
    get_near_seq(
        seq_fre_dir,
        s1_s2_selected_path,
        model.encoder,
        tokenizer,
        output_dir=s3_seqs_path
    )

    print(f"Results saved to {cfg.base.output_dir}")


if __name__ == "__main__":
    main()

    


