import os
import csv
import argparse
import torch



def get_args():
    parser = argparse.ArgumentParser(description="HC-HEBO for sequence generation")

    parser.add_argument("--SELEX_path", type=str, default=None, help="SELEX unique sequences file path, format: one unique sequence per line")
    parser.add_argument("--seq_act_path", type=str, default="/home/hanlab/2022_zzm/InstructNA_new/data/function_SELEX/IFN-gama/functional_label/label_data.csv", help="Sequence active file path, format: seq,activity")

    parser.add_argument("--encoder_model_path", type=str, default="/home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/function_SELEX/IFN-gama/encoder", help="Path to encoder model checkpoint")
    parser.add_argument("--decoder_model_path", type=str, default="/home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/function_SELEX/IFN-gama/decoder", help="Path to decoder model checkpoint")
    parser.add_argument("--BO_output_dir", type=str, default="./", help="Output directory for BO results")
    
    
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument("--search_r", type=float, default=5.0, help="Initial search radius")
    parser.add_argument("--max", action='store_true', help="Maximize or minimize the objective")
    parser.add_argument("--use_filter", action='store_true', help="Use filter for sequence selection")

    parser.add_argument("--HC_HEBO_batchsize", type=int, default=2, help="Number of sequences to generate per BO cycle")
    parser.add_argument("--model_down_dim", type=int, default=8, help="Dimension of reduced latent space")
    
    parser.add_argument("--f_primer", type=str, default="CGGTTCAG", help="5'primer")
    parser.add_argument("--r_primer", type=str, default="CTGAACCG", help="3'primer")

    
    return parser.parse_args()





if __name__ == "__main__":
    from InstructNA_frameworks.utills import set_seed,DNABERT_mask_seq_genernate_nolinkers,load_latest_checkpoint
    from InstructNA_frameworks.BO_utill import run_one_grouped_BO
    from InstructNA_frameworks.models.DNABERT_3mers.model import InstructNA
    from InstructNA_frameworks.batch_tokenize_func import DNABERT_3mer_tokenize_batch
    from transformers import AutoModel, AutoTokenizer

    args=get_args()
    seed_num=args.seed_num


    bo_batchsize=args.HC_HEBO_batchsize
    search_r=args.search_r
    model_down_dim=args.model_down_dim
    max=args.max
    use_filter=args.use_filter
    
    SELEX_path=args.SELEX_path
    BO_output_dir=args.BO_output_dir
    seq_act_path=args.seq_act_path
    encoder_model_path=args.encoder_model_path
    decoder_model_path=args.decoder_model_path
    
    f_primer=args.f_primer
    r_primer=args.r_primer

    
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


    SELEX_seqs=[]
    if SELEX_path is not None:
        with open(SELEX_path,"r") as f:
            lines=f.readlines()
            for line in lines:
                SELEX_seqs.append(line.strip())
            
    
    seeds_and_acts={}
    with open(seq_act_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        for row in reader:
            seeds_and_acts.update({row["seq"]:float(row["act"])}) 

    os.makedirs(BO_output_dir, exist_ok=True)
    

    all_seq_Kd=run_one_grouped_BO(
                            tokenizer,
                            encoder_model,
                            decoder_model,
                            device,
                            seeds_and_acts,
                            BO_output_dir,
                            bo_batchsize,
                            model_down_dim,
                            search_r,
                            SELEX_seqs,
                            max,
                            tokenize_function=DNABERT_3mer_tokenize_batch,
                            decode_func=DNABERT_mask_seq_genernate_nolinkers,
                            f_linker=f_primer,
                            r_linker=r_primer,
                            use_check_conditions=use_filter
                        )
    
        
