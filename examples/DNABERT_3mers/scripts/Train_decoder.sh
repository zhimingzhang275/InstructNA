python ../Train.py \
--dataset_dir /home/hanlab/2022_zzm/InstructNA_new/data/PTK7_scaffold_DNA_SELEX/SELEX_unique_seqs.csv \
--checkpoint_save_path /home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/PTK7/decoder \
--train_encoder_or_decoder 'train_decoder' \
--fintune_encoder_weight_path /home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/PTK7/encoder
