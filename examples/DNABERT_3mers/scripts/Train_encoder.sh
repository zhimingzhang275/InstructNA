python ../Train.py \
--dataset_dir /home/hanlab/2022_zzm/InstructNA_new/data/PTK7_scaffold_DNA_SELEX/R1_R2_R3_seq.txt \
--checkpoint_save_path /home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/PTK7_classifier \
--train_encoder_or_decoder 'train_encoder' \
--batchsize 64 \
--dataloader_num_worker 8 \
--evaluate_per_step 5 \
--train_epoches 100