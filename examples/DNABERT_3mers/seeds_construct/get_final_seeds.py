import os
import torch
import argparse
from InstructNA_frameworks.utills import set_seed,select_max_activity_by_cluster
from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
from transformers import AutoModel, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_final_seeds(seqs_act_dir,encode_model=None,tokenizer=None,output_dir=None):

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

    output_dir=os.path.join(output_dir, "Final_seeds.csv")
    final_seeds_seq,final_seeds_center_act=select_max_activity_by_cluster(seqs_act, encode_model, tokenizer,output_path=output_dir)
    
    return final_seeds_seq


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Construct final seeds from sequence activities")


    parser.add_argument("--seq_acts", type=str, default=None, help="Path to sequence activities CSV file,format: seq,activity") 
    parser.add_argument("--encoder_model_path", type=str, default=None, help="Path to encoder model checkpoint")
    parser.add_argument("--decoder_model_path", type=str, default=None,help="Path to decoder model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./", help="Output directory for BO results")
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument("--model_down_dim", type=int, default=8, help="Dimension of reduced latent space")
    

    return parser.parse_args()




if __name__ == "__main__":
    
    args=get_args()
    seed_num=args.seed_num
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
    
    get_final_seeds(seqs_act,encoder_model,tokenizer,output_dir=output_dir)

    


