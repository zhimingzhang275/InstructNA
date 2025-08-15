
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from InstructNA_frameworks.utills import   DNA_score_compute
import copy
from InstructNA_frameworks.filter_condition import check_conditions
import os

def HEBO_suggest(seq_KD, seq_embedding=None, max=True, num_to_gen=100, 
                 search_r=5, center_based=True, lower_bound_tensor=None, upper_bound_tensor=None):

    embeds = seq_embedding
    X_init = []
    Y_init = []  
    seqs_X_Y = {}

    for seq in seq_KD.keys():
        X_init.append(embeds[seq].numpy().flatten())
        if max:
            Y_init.append(-float(seq_KD[seq]))
        else:
            Y_init.append(float(seq_KD[seq]))
        seqs_X_Y[seq] = list([embeds[seq].numpy(), float(seq_KD[seq])])

    hidden_size = embeds[seq].view(-1).shape[0]
    columns_list = ["x" + str(i) for i in range(hidden_size)]
    X_init = pd.DataFrame(np.array(X_init), columns=columns_list)
    Y_init = np.array(Y_init).reshape(-1, 1)


    params = []
    for i in range(hidden_size):
        if lower_bound_tensor is not None and upper_bound_tensor is not None and not center_based:
    
            lb = lower_bound_tensor[i].item()
            ub = upper_bound_tensor[i].item()
        elif center_based:
            
            lb = np.min(X_init.iloc[:, i]) - search_r
            ub = np.max(X_init.iloc[:, i]) + search_r
        else:
            raise ValueError("Specify bounds or enable center_based mode.")
        params.append({'name': f"x{i}", 'type': 'num', 'lb': lb, 'ub': ub})

    space = DesignSpace().parse(params)

    opt = HEBO(space, model_name="gp")
    opt.observe(X_init, Y_init)

    rec = opt.suggest(n_suggestions=num_to_gen)
    candidates_list = [rec.iloc[i, :] for i in range(rec.shape[0])]

    return candidates_list






def run_grouped_BO(
    tokenizer,
    encode_model,
    decode_model,
    device,
    BO_cycle_nums,
    Kd_seq_path_or_dict,
    output_dir,
    label_dir,
    bo_batchsize,
    model_down_dim,
    init_search_r,
    min_r,
    f_linker,
    r_linker,
    resume_path=None,
    tokenize_function=None,
    decode_func=None,
):  
    cur_search_r=init_search_r
    generated_seq_set = set()
    group_affinities = {}  # {init_seq: [[seq1, score1], [seq2, score2], ...]}
    completed_steps = 0
    all_gen_seq_act = {}


    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path} ...")
        with open(resume_path, "r") as f:
            for line in f:
                init_seq, gen_seq, score = line.strip().split(",")
                score = float(score)
                if init_seq not in group_affinities:
                    group_affinities[init_seq] = []
                group_affinities[init_seq].append([gen_seq, score])
                generated_seq_set.add(gen_seq)

        completed_steps = (len(next(iter(group_affinities.values()))) - 1) // bo_batchsize
        print(f"Detected completed BO steps: {completed_steps}\n")

    else:
        print("No resume path provided or file does not exist. Starting from scratch.")
        if isinstance(Kd_seq_path_or_dict, dict):
            for seq, kd in Kd_seq_path_or_dict.items():
                kd = float(kd)
                generated_seq_set.add(seq)
                group_affinities[seq] = [[seq, kd]]
        elif isinstance(Kd_seq_path_or_dict, str) and os.path.exists(Kd_seq_path_or_dict):
            with open(Kd_seq_path_or_dict, "r") as f:
                for line in f:
                    seq, kd = line.strip().split(",")
                    kd = float(kd)
                    generated_seq_set.add(seq)
                    group_affinities[seq] = [[seq, kd]]


    for step in tqdm(range(completed_steps, BO_cycle_nums), desc="BO_step"):
        new_group_affinities = {}
        
        if step > 0 and cur_search_r > min_r:
            cur_search_r *= 0.5
        elif cur_search_r > min_r and cur_search_r / 2 < min_r:
            cur_search_r = min_r
        print(f"Current search radius is {cur_search_r}\n")

        for init_seq, records in group_affinities.items():
            print(f"[STEP {step}] Group: {init_seq}")

           
            best_seq,best_act = max(records, key=lambda x: x[1]) if max else min(records, key=lambda x: x[1])

            seq_KD = {best_seq: best_act}
            examples = torch.tensor(tokenize_function([best_seq],tokenizer),device=device)
            with torch.no_grad():
                output = encode_model(examples)
            embedding = output.cpu()
            seq_embedding = {best_seq: embedding}


            next_point_list = HEBO_suggest(seq_KD, seq_embedding, num_to_gen=bo_batchsize, search_r=cur_search_r)

            
            generated_seqs = []
            for point in tqdm(next_point_list, desc=f"[Step {step}] Gen for group {init_seq}"):
                point_tensor = torch.tensor(point, dtype=torch.float32).to(device).reshape(1, -1, model_down_dim)
                with torch.no_grad():
                    
                    generated_seq = decode_func(point_tensor, decode_model, encode_model,tokenizer,pre_link=f_linker, rever_link=r_linker)
                    if not generated_seq:
                        continue

                    if f_linker and r_linker:
                        core_seq = generated_seq[len(f_linker):-len(r_linker)]
                    else:
                        core_seq = generated_seq
                    unique_gen_tries = 0
                    sample_T=1
                    while core_seq in generated_seq_set:
                        print(f"Duplicate {core_seq}, {unique_gen_tries}st retrying...")
                        tmp_cur_search_r = 1.05 ** (unique_gen_tries+1) * cur_search_r
                        sample_T=sample_T+0.05
                        print(f"Current search radius is {tmp_cur_search_r}\n")
                        next_point = HEBO_suggest(seq_KD, seq_embedding, num_to_gen=1,search_r=tmp_cur_search_r)[0]
                        point_tensor = torch.tensor(next_point, dtype=torch.float32).to(device).reshape(1, -1, model_down_dim)
                        generated_seq = decode_func(point_tensor, decode_model,encode_model,  tokenizer,
                                                                    pre_link=f_linker, rever_link=r_linker,temperature=sample_T)
                        if f_linker and r_linker:
                            core_seq = generated_seq[len(f_linker):-len(r_linker)]
                        else:
                            core_seq = generated_seq
                        unique_gen_tries += 1

                    if core_seq not in generated_seq_set:
                        score = DNA_score_compute([generated_seq], score_file=label_dir)[0]
                        generated_seq_set.add(core_seq)
                        generated_seqs.append([core_seq, score])
                        all_gen_seq_act[core_seq] = score
                        
            new_group_affinities[init_seq] = records + generated_seqs

        group_affinities = new_group_affinities

    output_path = os.path.join(output_dir, "HC-HEBO_gen_seq.csv")
    with open(output_path, "w") as f:
        f.write("init_seq,gen_seq,act\n")
        for init_seq, records in group_affinities.items():
            for seq, score in records:
                f.write(f"{init_seq},{seq},{score}\n")

    print("✅ Grouped BO completed.")
    
    return all_gen_seq_act, group_affinities







def run_one_grouped_BO(
    tokenizer,
    encode_model,
    decode_model,
    device,
    Kd_seq_path_or_dict,
    output_dir,
    bo_batchsize,
    model_down_dim,
    search_r,
    SELEX_seqs=None,
    max_object=True,
    tokenize_function=None,
    decode_func=None,
    f_linker=None,
    r_linker=None,
    use_check_conditions=False

):  
    cur_search_r=search_r
    generated_seq_set = set()
    group_affinities = {}  # {init_seq: [[seq1, score1], [seq2, score2], ...]}



    if isinstance(Kd_seq_path_or_dict, dict):
        for seq, kd in Kd_seq_path_or_dict.items():
            kd = float(kd)
            generated_seq_set.add(seq)
            group_affinities[seq] = [[seq, kd]]
    elif isinstance(Kd_seq_path_or_dict, str) and os.path.exists(Kd_seq_path_or_dict):
        with open(Kd_seq_path_or_dict, "r") as f:
            for line in f:
                seq, kd = line.strip().split(",")
                kd = float(kd)
                generated_seq_set.add(seq)
                group_affinities[seq] = [[seq, kd]]


    new_group_affinities = {}
    

    for init_seq, records in group_affinities.items():
        print(f"Group init sequence: {init_seq}")

        best_seq, best_score = max(records, key=lambda x: x[1])
        seq_KD = {best_seq: best_score}

        examples = torch.tensor(tokenize_function([best_seq],tokenizer),device=device)

        with torch.no_grad():
            output = encode_model(examples)
        embedding = output.cpu()[0]
        seq_embedding = {best_seq: embedding}

        # BO
        next_point_list =HEBO_suggest(seq_KD, seq_embedding, num_to_gen=bo_batchsize, search_r=cur_search_r,max=max_object)

        generated_seqs = []
        for point in tqdm(next_point_list, desc=f"[Generation for group {init_seq}"):
            point_tensor = torch.tensor(point, dtype=torch.float32).to(device).reshape(1, -1, model_down_dim)
            with torch.no_grad():

                generated_seq = decode_func(point_tensor, decode_model, encode_model, tokenizer,
                                                            pre_link=f_linker, rever_link=r_linker)
                if not generated_seq:
                    continue
                
                if f_linker and r_linker:
                    core_seq = generated_seq[len(f_linker):-len(r_linker)]
                else:
                    core_seq = generated_seq
                unique_gen_tries = 0
                sample_T=1
                secondary_structure = None 
                while (
                        (core_seq in generated_seq_set)
                        or (core_seq in SELEX_seqs)
                        or (use_check_conditions and not (secondary_structure := check_conditions(generated_seq)))
                    ):
                    if core_seq in SELEX_seqs:  
                        print(f"{core_seq} in SELEX sequence, skipping...")

                    if core_seq in generated_seq_set:
                        
                        tmp_cur_search_r = 1.05 ** (unique_gen_tries+1) * cur_search_r
                        sample_T=sample_T+0.05
                        print(f"Duplicate {core_seq}, {unique_gen_tries}st retrying...")
                        print(f"Increasing search radius from {cur_search_r:.4f} to {tmp_cur_search_r:.4f}")
                        print(f"Increasing sampling temperature to {sample_T:.2f}")
                    if not check_conditions(generated_seq) and not (core_seq in generated_seq_set):
                        unique_gen_tries=0
                        tmp_cur_search_r = cur_search_r
                        sample_T=1
                    next_point = HEBO_suggest(seq_KD, seq_embedding, num_to_gen=1,search_r=cur_search_r,max=max_object)[0]
                    point_tensor = torch.tensor(next_point, dtype=torch.float32).to(device).reshape(1, -1, model_down_dim)
                    generated_seq = decode_func(point_tensor, decode_model,  encode_model, tokenizer,
                                                                pre_link=f_linker, rever_link=r_linker,temperature=sample_T)
                    if f_linker and r_linker:
                        core_seq = generated_seq[len(f_linker):-len(r_linker)]
                    else:
                        core_seq = generated_seq
                    unique_gen_tries += 1
                
                
                generated_seq_set.add(core_seq)
                generated_seqs.append([core_seq,secondary_structure]) 
                print(f"Gen sequence: {core_seq}")
  
            
        new_group_affinities[init_seq] = records + generated_seqs

    group_affinities = new_group_affinities


    output_path = os.path.join(output_dir, "HC-HEBO_gen_seq.csv")
    with open(output_path, "w") as f:
        f.write("init_seq,gen_seq,act_or_ss\n")
        for init_seq, records in group_affinities.items():
            for seq, score in records:
                f.write(f"{init_seq},{seq},{score}\n")

    print(f"✅ Grouped BO completed and write into {output_path} .")
    
    return  group_affinities




def run_global_BO( tokenizer, 
                  encode_model, 
                  decode_model, 
                  device, 
                  BO_cycle_nums,
                  Kd_seq_path_or_dict, 
                  search_bound, 
                  bo_batchsize, 
                  f_linker,
                  r_linker,
                  label_dir,
                  model_down_dim,
                  BO_output_dir="./",
                  tokenize_function=None,
                  decode_func=None
                  ):

    generated_seq_set = set()
    seq_KD = {}           # {seq: score}
    seq_embedding = {}    # {seq: embedding}
    
    gen_seq_act={}
    step_seq_KD = []
    
    if isinstance(Kd_seq_path_or_dict, dict):
        seq_KD=Kd_seq_path_or_dict
        for seq, kd in Kd_seq_path_or_dict.items():
            generated_seq_set.add(seq)
    else:
        with open(Kd_seq_path_or_dict, "r") as f:
            for line in f:
                seq, kd = line.strip().split(",")
                kd = float(kd)
                seq_KD[seq] = kd
                generated_seq_set.add(seq)
                
    
    for step in tqdm(range(BO_cycle_nums), desc="Global_BO_Step"):
        print(f"\n[Global BO] Step {step}\n")

                 
        seq_embedding.clear()
        for seq in seq_KD:
            example = torch.tensor(tokenize_function([seq],tokenizer), device=device)
            with torch.no_grad():
                output = encode_model(example)[0]
            embedding = output.cpu()
            seq_embedding[seq] = embedding

        BO_seq_KD=copy.deepcopy(seq_KD)  
        
        next_points = HEBO_suggest(BO_seq_KD, seq_embedding, num_to_gen=bo_batchsize, center_based=False, upper_bound_tensor=search_bound["search_up_bound"], lower_bound_tensor=search_bound["search_lower_bound"])


        # ===== Decoding + Scoring + Deduplication =====
        for point in tqdm(next_points, desc=f"[Step {step}] Decoding"):
            point_tensor = torch.tensor(point, dtype=torch.float32).to(device).reshape(1, -1, model_down_dim)
            with torch.no_grad():
                generated_seq = decode_func(point_tensor, decode_model, encode_model, tokenizer,
                                                                pre_link=f_linker, rever_link=r_linker)
                if not generated_seq:
                    continue

                if f_linker and r_linker:
                    core_seq = generated_seq[len(f_linker):-len(r_linker)]
                else:
                    core_seq = generated_seq
                    
                unique_gen_tries = 0
                while core_seq in generated_seq_set:
                    print(f"Duplicate {core_seq}, {unique_gen_tries}st retrying...")

                    retry_point = HEBO_suggest(BO_seq_KD, seq_embedding, num_to_gen=bo_batchsize, center_based=False, upper_bound_tensor=search_bound["search_up_bound"], lower_bound_tensor=search_bound["search_lower_bound"])[0]
                    point_tensor = torch.tensor(retry_point, dtype=torch.float32).to(device).reshape(1, -1, model_down_dim)
                    generated_seq = decode_func(point_tensor, decode_model,  encode_model,tokenizer,
                                                                pre_link=f_linker, rever_link=r_linker)
                    if not generated_seq:
                        break

                    if f_linker and r_linker:
                        core_seq = generated_seq[len(f_linker):-len(r_linker)]
                    else:
                        core_seq = generated_seq
                    unique_gen_tries += 1


                if core_seq not in generated_seq_set:
                    score = DNA_score_compute([generated_seq], score_file=label_dir)[0]
                    seq_KD[core_seq] = score
                    gen_seq_act[core_seq] = score
                    generated_seq_set.add(core_seq)
                    step_seq_KD.append([step,core_seq,score])
    
    with open(os.path.join(BO_output_dir, "HEBO_round_seq_act.csv"), "w") as f:
        f.write("round,seq,act\n")
        for round, seq, activity in step_seq_KD:
            f.write(f"{round},{seq},{activity}\n")
                    

    print("✅ Global BO finished.")

    return gen_seq_act, step_seq_KD


