
import argparse
import os
import torch
from InstructNA_frameworks.utills import set_seed,get_cluster_center_seqs
from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
from transformers import AutoModel, AutoTokenizer
import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser(description="Generate GMM sequence")

    parser.add_argument("--sequences_dir", type=str, default=None, help="Training sequences file. Path to a CSV file containing sequences, one per line")
    parser.add_argument("--tSNE_visual", type=bool, default=False, help="Whether to visualize t-SNE results")
    parser.add_argument("--encoder_model_path", type=str, default=None, help="Path to encoder model checkpoint")
    parser.add_argument("--decoder_model_path", type=str, default=None, help="Path to decoder model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./", help="Output directory for BO results")
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument("--model_down_dim", type=int, default=8, help="Dimension of reduced latent space")
    
    return parser.parse_args()




if __name__ == "__main__":
    
    
    args=get_args()
    seed_num=args.seed_num
    model_down_dim=args.model_down_dim
    BO_output_dir=args.output_dir
    data_dir=args.sequences_dir
    encoder_model_path=args.encoder_model_path
    decoder_model_path=args.decoder_model_path
    visual_tSNE=args.tSNE_visual

    
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

    generated_seqs,_,_,_=get_cluster_center_seqs(data_dir, encoder_model, decoder_model, tokenizer,tSNE=visual_tSNE,model_down_dim=model_down_dim)
    
    with open(os.path.join(BO_output_dir, "GMM_center_seqs.csv"), "w") as f:
        for seq in generated_seqs:
            print(f"GMM generated sequence: {seq}")
            f.write(f"{seq}\n")
        print(f"Generated sequences saved to {os.path.join(BO_output_dir, 'GMM_center_seqs.txt')}")
        
        