python ../single_HC-HEBO_inference.py \
    --SELEX_path /home/hanlab/2022_zzm/InstructNA_new/data/Thrombin_scaffold_DNA_SELEX/SELEX_unique_seqs.csv \
    --seq_act_path /home/hanlab/2022_zzm/InstructNA_new/data/Thrombin_scaffold_DNA_SELEX/Seeds_act.csv \
    --encoder_model_path /home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/Thrombin_scaffold_DNA_SELEX/encoder\
    --decoder_model_path /home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/Thrombin_scaffold_DNA_SELEX/decoder \
    --BO_output_dir ./ \
    --search_r 5 \
    --max  \
    --use_filter \
    --HC_HEBO_batchsize 10 \
    --model_down_dim 8 \
    --f_primer ATCTAAC \
    --r_primer CGGTTAGA
 # -m debugpy --listen localhost:5678 --wait-for-client 