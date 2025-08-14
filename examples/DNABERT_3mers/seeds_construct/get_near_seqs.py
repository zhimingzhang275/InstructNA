
import argparse
import os
import torch
from InstructNA_frameworks.utills import set_seed,kmers_sliding_windows
from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
from transformers import AutoModel, AutoTokenizer
from InstructNA_frameworks.utills import kmers_sliding_windows,select_max_activity_by_cluster\
    ,encode_sequences_to_embeddings,find_nearest_neighbor


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
        while (chosen_seq_list[idx][0] in seeds or chosen_seq_list[idx][0] in s3) :
            idx+=1

        s3.append(chosen_seq_list[idx][0])
        
    seeds.update(s3)
    
    
    for seq in s3:
        with open(os.path.join(output_dir, "find_near_seqs_in_SELEX.csv"), "a") as f:
            f.write(f"{seq}\n")

    return s3


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Get near sequences from SELEX data")

    parser.add_argument("--seqs_fre_dir", type=str, default=None, help="sequence frequency csv file path")
    parser.add_argument("--seq_acts", type=str, default=None, help="sequence activities csv file path, format: seq,activity")
    parser.add_argument("--encoder_model_path", type=str, default=None, help="Path to encoder model checkpoint")
    parser.add_argument("--decoder_model_path", type=str, default=None, help="Path to decoder model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./", help="Output directory for BO results")
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument("--model_down_dim", type=int, default=8, help="Dimension of reduced latent space")
    

    return parser.parse_args()




if __name__ == "__main__":
    
    args=get_args()
    seed_num=args.seed_num
    seqs_fre=args.seqs_fre_dir
    seqs_act=args.seq_acts
    model_down_dim=args.model_down_dim
    output_dir=args.output_dir
    encoder_model_path=args.encoder_model_path
    decoder_model_path=args.decoder_model_path
    
    set_seed(seed_num)
    print("Loading LM model...")

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
    
    print(model.config)
    
    lm_embeds_dim = model.config.hidden_size
    lm_vocab_size = model.config.vocab_size
    full_model=InstructNA(model,lm_embeds_dim,lm_vocab_size,proj_dim=model_down_dim)

    decoder_model = full_model.decoder
    encoder_model = full_model.encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_model.to(device)
    decoder_model.to(device)
    full_model.to(device)
    
    get_near_seq(seqs_fre,seqs_act,encoder_model,tokenizer,output_dir=output_dir)

    


