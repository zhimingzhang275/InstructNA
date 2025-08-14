import os
import re
from collections import Counter
import torch
import pandas as pd
import argparse
from InstructNA_frameworks.BO_utill import run_global_BO, run_grouped_BO

from InstructNA_frameworks.utills import (
    set_seed,
    calculate_similarities_needle,
    load_latest_checkpoint,
    DNA_score_compute,
    DNABERT_mask_seq_genernate_nolinkers,
    kmers_sliding_windows,
    select_max_activity_by_cluster,
    get_cluster_center_seqs,
    find_nearest_neighbor,
    get_cluster_center_seqs
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_fastq_filename(filename):
    # 去掉路径，只保留文件名
    base = os.path.basename(filename)
    # 去掉扩展名
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
    



def get_top_sequences_from_fastq(fastq_path: str, top_n: int = 10) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Extract the top N most frequent sequences from a FASTQ file.

    Args:
        fastq_path (str): Path to the FASTQ file.
        top_n (int): Number of top sequences to extract, sorted by frequency.

    Returns:
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
            - List of the top N sequences with their counts [(sequence, count), ...]
            - List of all sequences with their counts [(sequence, count), ...]
    """
    sequence_counter = Counter()
    
    with open(fastq_path, "r") as f:
        line_idx = 0
        for line in f:
            if line_idx % 4 == 1:  # Sequence lines in FASTQ appear every 4 lines
                seq = line.strip()
                sequence_counter[seq] += 1
            line_idx += 1

    return sequence_counter.most_common(top_n), sequence_counter.most_common()




def get_seeds_seq(fastq_dir,label_dir=None,encode_model=None,decode_model=None,tokenizer=None,output_dir=None):
    seeds=set()
    device=next(encode_model.parameters()).device
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
    s2,cat_seq_embeds, all_sequence_embed_dict,all_embeds_seq_dict=get_cluster_center_seqs(fastq_dir,  encode_model, decode_model, tokenizer)
    seeds.update(s2)
    #Source 3 seeds

    full_len_seqs=[]
    for seq in seeds:
        full_len_seqs.append(fastq_info["left_flank"].strip()+seq.strip()+fastq_info["right_flank"].strip())
    scores=DNA_score_compute(full_len_seqs,score_file=label_dir)
    s1_s2_seq_score=dict(zip(seeds, scores))
    
    s3_center_seq,s3_center_act=select_max_activity_by_cluster(s1_s2_seq_score, encode_model, tokenizer)
    
    
    mers_sep_seq=[kmers_sliding_windows(seq) for seq in s3_center_seq]
    input_ids = tokenizer.batch_encode_plus(mers_sep_seq, add_special_tokens=True, max_length=512)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device)

    with torch.no_grad():
        output = encode_model(input_ids)
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
    
    
    final_seeds,final_seeds_act=select_max_activity_by_cluster(s1_s2_s3_seq_score, encode_model, tokenizer,output_path=os.path.join(output_dir,"final_seeds.csv"), verbose=True)
    
    
    with open(os.path.join(output_dir, "s1_s2_s3.csv"), "w") as f:
        f.write(f"Source,Sequence,Act\n")
        for seq in s1:
            f.write(f"s1,{seq},{s1_s2_s3_seq_score[seq]}\n")
        for seq in s2:
            f.write(f"s2,{seq},{s1_s2_s3_seq_score[seq]}\n")
        for seq in s3:
            f.write(f"s3,{seq},{s1_s2_s3_seq_score[seq]}\n")

    return dict(zip(final_seeds, final_seeds_act)),fastq_info


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="pipeline for TFs InstructNA with DNABERT_3mers")
    
    parser.add_argument("--fastq_dir", type=str, default=None, help="public TFs SELEX fastaq file path")
    parser.add_argument("--BO_type", type=str, default="HEBO", help="Bayesian Optimization type, options: HC-HEBO, HEBO")
    parser.add_argument("--label_dir", type=str, default=None, help="Path to label or score file (raw/motif)")
    parser.add_argument("--encoder_model_path", type=str, default=None, help="Path to encoder model checkpoint")
    parser.add_argument("--decoder_model_path", type=str, default=None, help="Path to decoder model checkpoint")
    parser.add_argument("--BO_output_dir", type=str, default="./", help="Output directory for BO results")

    
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument("--init_search_r", type=float, default=5.0, help="HC-HEBO initial search radius")
    parser.add_argument("--min_r", type=float, default=1.25, help="HC-HEBO minimum search radius")
    parser.add_argument("--BO_cycle_nums", type=int, default=10, help="Number of BO iterations")
    parser.add_argument("--bo_batchsize", type=int, default=10, help="Number of sequences to generate per BO cycle")
    parser.add_argument("--model_down_dim", type=int, default=8, help="Dimension of reduced latent space")
    
    parser.add_argument("--resume_path", type=str, default=None, help="Optional resume CSV path")

    return parser.parse_args()




if __name__ == "__main__":
    
    from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
    from InstructNA_frameworks.batch_tokenize_func import DNABERT_3mer_tokenize_batch
    from InstructNA_frameworks.utills import set_seed,load_latest_checkpoint,get_cluster_center_sequence_embeddings
    from transformers import AutoModel, AutoTokenizer
    args=get_args()
    seed_num=args.seed_num
    BO_type=args.BO_type
    BO_cycle_nums=args.BO_cycle_nums
    bo_batchsize=args.bo_batchsize
    init_search_r=args.init_search_r
    min_r=args.min_r
    model_down_dim=args.model_down_dim
    
    BO_output_dir=args.BO_output_dir
    fastaq_dir=args.fastq_dir
    encoder_model_path=args.encoder_model_path
    decoder_model_path=args.decoder_model_path
    label_dir=args.label_dir
    
    set_seed(seed_num)
    print("Loading LM model...")
    # config = AutoConfig.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    
    
    print(model.config)
    
    lm_embeds_dim = model.config.hidden_size
    lm_vocab_size = model.config.vocab_size
    full_model=InstructNA(model,lm_embeds_dim,lm_vocab_size,proj_dim=model_down_dim)

    full_model.encoder, _ = load_latest_checkpoint(full_model.encoder,encoder_model_path)
    full_model.decoder, _ = load_latest_checkpoint(full_model.decoder,decoder_model_path)

    decoder_model = full_model.decoder
    encoder_model = full_model.encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"


    encoder_model.to(device)
    decoder_model.to(device)
    full_model.to(device)

    
    seeds_and_acts,fastq_info=get_seeds_seq(fastaq_dir,label_dir,encoder_model,decoder_model,tokenizer,output_dir=BO_output_dir)
    sim_compute_ref_seq=list(seeds_and_acts.keys())
    
    BO_output_dir=os.path.join(BO_output_dir, fastq_info["transcription_factor"])
    os.makedirs(BO_output_dir, exist_ok=True)
    
    if BO_type=="HC-HEBO":
        bo_batchsize=int(bo_batchsize/len(seeds_and_acts))
        all_seq_Kd, _= run_grouped_BO(
                                tokenizer,
                                encoder_model,
                                decoder_model,
                                device,
                                BO_cycle_nums,
                                seeds_and_acts,
                                BO_output_dir,
                                label_dir,
                                bo_batchsize,
                                model_down_dim,
                                init_search_r,
                                min_r,
                                fastq_info["left_flank"],
                                fastq_info["right_flank"],
                                tokenize_function=DNABERT_3mer_tokenize_batch,
                                decode_func=DNABERT_mask_seq_genernate_nolinkers
                            )
        
    
    elif BO_type=="HEBO":

            cluster_center_emb, sequence_embedding, all_batch_sequence_dict,all_batch_tensor_seq_dict=get_cluster_center_sequence_embeddings(
                encoder_model=encoder_model,
                tokenizer=tokenizer,
                data_path=fastaq_dir,
            )
            
            search_up_bound = sequence_embedding.max(dim=0).values
            search_lower_bound = sequence_embedding.min(dim=0).values
            search_bound={"search_up_bound":search_up_bound,"search_lower_bound":search_lower_bound}
            all_seq_Kd, _ =run_global_BO(
                                        tokenizer,
                                        encoder_model,
                                        decoder_model,
                                        device=device,
                                        search_bound=search_bound,
                                        BO_cycle_nums=BO_cycle_nums,
                                        Kd_seq_path_or_dict=seeds_and_acts,
                                        BO_output_dir=BO_output_dir,
                                        label_dir=label_dir,
                                        bo_batchsize=bo_batchsize,
                                        model_down_dim=model_down_dim,
                                        f_linker=fastq_info["left_flank"],
                                        r_linker=fastq_info["right_flank"],
                                        tokenize_function=DNABERT_3mer_tokenize_batch,
                                        decode_func=DNABERT_mask_seq_genernate_nolinkers
                                    )

    gen_seq,act=list(all_seq_Kd.keys()),list(all_seq_Kd.values())                                    
    sim_results=calculate_similarities_needle(sim_compute_ref_seq, gen_seq,)
    gen_seq_sim_act_results=[[i[0][0],i[0][1],i[1]] for i in list(zip(sim_results, act, )) ]
    
    with open(os.path.join(BO_output_dir, f"{BO_type}_seq_sim_act.csv"), "w") as f:
        f.write("Generated_Sequence,Similarity,Activity,\n")
        for seq, activity, similarity in gen_seq_sim_act_results:
            f.write(f"{seq},{activity},{similarity}\n")
                