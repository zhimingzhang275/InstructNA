import os
import re
from collections import Counter
import torch
import pandas as pd
from InstructNA_frameworks.BO_utill import run_global_BO, run_grouped_BO
import hydra
from omegaconf import DictConfig
from InstructNA_frameworks.utills import (
    set_seed,
    calculate_similarities_needle,
    DNA_score_compute,
    DNABERT_mask_seq_generate_nolinkers,
    kmers_sliding_windows,
    select_max_activity_by_cluster,
    get_cluster_center_seqs,
    find_nearest_neighbor,
    get_cluster_center_seqs
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_fastq_filename(filename):
    base = os.path.basename(filename)
    name = base.replace('.fastq', '')
    parts = name.split('_')

    if len(parts) < 4:
        raise ValueError("文件名格式不正确，应至少包含4个下划线分隔字段")

    tf_name = parts[0]
    seq_region = parts[1]
    experiment_type = parts[2]
    round = parts[3]

    # 尝试进一步解析中间那段，例如 TGCCGC20NGA
    match = re.match(r"([ACGT]*)(\d+)N([ACGT]*)", seq_region)
    if match:
        left_flank, num_N, right_flank = match.groups()
    else:
        left_flank, num_N, right_flank = None, None, None

    return {
        "transcription_factor": tf_name,
        "full_variable_region": seq_region,
        "left_flank": left_flank,
        "num_random_N": int(num_N) if num_N else None,
        "right_flank": right_flank,
        "experiment_type": experiment_type,
        "round": round
    }
    



def get_top_sequences_from_fastq(fastq_path: str, top_n: int = 10): 
    
    sequence_counter = Counter()
    
    with open(fastq_path, "r") as f:
        line_idx = 0
        for line in f:
            if line_idx % 4 == 1:  # Sequence lines in FASTQ appear every 4 lines
                seq = line.strip()
                sequence_counter[seq] += 1
            line_idx += 1

    return sequence_counter.most_common(top_n), sequence_counter.most_common()



from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
def get_seeds_seq(fastq_dir,label_dir=None,InstructNA_model:InstructNA =None,tokenizer=None,output_dir=None):
    seeds=set()
    device=next(InstructNA_model.parameters()).device
    file_name=os.path.basename(fastq_dir)
    fastq_info=parse_fastq_filename(file_name)
    output_dir=os.path.join(output_dir, fastq_info["transcription_factor"])
    os.makedirs(output_dir, exist_ok=True)
    final_seeds_path=os.path.join(output_dir,"final_seeds.csv")
    if os.path.exists(final_seeds_path):
        print(f"[INFO] Detected existing {final_seeds_path}, loading directly...")
        df = pd.read_csv(final_seeds_path)
        final_seeds = list(df["Sequence"])
        final_acts = list(df["Activity"])
        return dict(zip(final_seeds, final_acts)), fastq_info
    
    #Source 1 seeds
    s1,all_seq_fre_dict=get_top_sequences_from_fastq(fastq_dir, top_n=10)
    s1=[i[0] for i in s1]
    all_seq_fre_dict=dict(all_seq_fre_dict)
    seeds.update(s1)
    #Source 2 seeds  
    s2,cat_seq_embeds, all_sequence_embed_dict,all_embeds_seq_dict=get_cluster_center_seqs(fastq_dir,  InstructNA_model, tokenizer,model_down_dim=InstructNA_model.config.proj_dim)
    seeds.update(s2)
    #Source 3 seeds

    full_len_seqs=[]
    for seq in seeds:
        full_len_seqs.append(fastq_info["left_flank"].strip()+seq.strip()+fastq_info["right_flank"].strip())
    scores=DNA_score_compute(full_len_seqs,score_file=label_dir)
    s1_s2_seq_score=dict(zip(seeds, scores))
    
    s3_center_seq,s3_center_act=select_max_activity_by_cluster(s1_s2_seq_score, InstructNA_model.encoder, tokenizer)
    
    
    mers_sep_seq=[kmers_sliding_windows(seq) for seq in s3_center_seq]
    input_ids = tokenizer.batch_encode_plus(mers_sep_seq, add_special_tokens=True, max_length=512)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device)

    with torch.no_grad():
        output = InstructNA_model.encoder(input_ids)["latent"]
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
        while (chosen_seq_list[idx][0] in seeds):
            idx+=1

        s3.append(chosen_seq_list[idx][0])
        
    seeds.update(s3)
    
    
    full_len_seqs=[]
    for seq in seeds:
        full_len_seqs.append(fastq_info["left_flank"].strip()+seq.strip()+fastq_info["right_flank"].strip())
    scores=DNA_score_compute(full_len_seqs,score_file=label_dir)
    s1_s2_s3_seq_score=dict(zip(seeds, scores))
    
    
    final_seeds,final_seeds_act=select_max_activity_by_cluster(s1_s2_s3_seq_score, InstructNA_model.encoder, tokenizer,output_path=output_dir, verbose=True)
    
    
    with open(os.path.join(output_dir, "s1_s2_s3.csv"), "w") as f:
        f.write(f"Source,Sequence,Act\n")
        for seq in s1:
            f.write(f"s1,{seq},{s1_s2_s3_seq_score[seq]}\n")
        for seq in s2:
            f.write(f"s2,{seq},{s1_s2_s3_seq_score[seq]}\n")
        for seq in s3:
            f.write(f"s3,{seq},{s1_s2_s3_seq_score[seq]}\n")

    return dict(zip(final_seeds, final_seeds_act)),fastq_info


@hydra.main(version_base=None, config_path="conf", config_name="TFs_pipeline")
def main(cfg: DictConfig):

    from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
    from InstructNA_frameworks.batch_tokenize_func import DNABERT_3mer_tokenize_batch
    from InstructNA_frameworks.utills import (
        set_seed,
        get_cluster_center_sequence_embeddings
    )

    import torch, os

    set_seed(cfg.seed_num)

    print("Loading InstructNA model...")
    model = InstructNA.from_pretrained(cfg.paths.model)
    tokenizer = model.tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    encoder_model = model.encoder
    decoder_model = model.decoder
    encoder_model.to(device)
    decoder_model.to(device)

    # === prepare data ===
    seeds_and_acts, fastq_info = get_seeds_seq(
        cfg.paths.fastq,
        cfg.paths.label,
        model,
        tokenizer,
        output_dir=cfg.paths.output
    )

    sim_compute_ref_seq = list(seeds_and_acts.keys())

    BO_output_dir = os.path.join(
        cfg.paths.output,
        fastq_info["transcription_factor"]
    )
    os.makedirs(BO_output_dir, exist_ok=True)

    # === BO logic ===
    if cfg.bo.type == "HC-HEBO":
        bo_batchsize = int(cfg.bo.batchsize / len(seeds_and_acts))

        all_seq_Kd, _ = run_grouped_BO(
            tokenizer,
            model,
            device,
            cfg.bo.cycle_nums,
            seeds_and_acts,
            BO_output_dir,
            cfg.paths.label,
            bo_batchsize,
            model.config.proj_dim,
            cfg.bo.init_search_r,
            cfg.bo.min_r,
            fastq_info["left_flank"],
            fastq_info["right_flank"],
            tokenize_function=DNABERT_3mer_tokenize_batch,
            decode_func=DNABERT_mask_seq_generate_nolinkers
        )

    elif cfg.bo.type == "HEBO":

        cluster_center_emb, sequence_embedding, *_ = (
            get_cluster_center_sequence_embeddings(
                encoder_model=encoder_model,
                tokenizer=tokenizer,
                data_path=cfg.paths.fastq,
            )
        )

        search_bound = {
            "search_up_bound": sequence_embedding.max(dim=0).values,
            "search_lower_bound": sequence_embedding.min(dim=0).values,
        }

        all_seq_Kd, _ = run_global_BO(
            tokenizer,
            model,
            device=device,
            search_bound=search_bound,
            BO_cycle_nums=cfg.bo.cycle_nums,
            Kd_seq_path_or_dict=seeds_and_acts,
            BO_output_dir=BO_output_dir,
            label_dir=cfg.paths.label,
            bo_batchsize=cfg.bo.batchsize,
            model_down_dim=model.config.proj_dim,
            f_linker=fastq_info["left_flank"],
            r_linker=fastq_info["right_flank"],
            tokenize_function=DNABERT_3mer_tokenize_batch,
            decode_func=DNABERT_mask_seq_generate_nolinkers
        )


    gen_seq = list(all_seq_Kd.keys())
    act = list(all_seq_Kd.values())

    sim_results = calculate_similarities_needle(sim_compute_ref_seq, gen_seq)

    with open(os.path.join(BO_output_dir, f"{cfg.bo.type}_seq_sim_act.csv"), "w") as f:
        f.write("Generated_Sequence,Similarity,Activity\n")
        for (sim_seq, sim), activity in zip(sim_results, act):
            f.write(f"{sim_seq},{sim},{activity}\n")

if __name__ == "__main__":


    main()

                