import csv
import copy
import os
import re
import tempfile
from collections import Counter, defaultdict
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from Bio.Emboss.Applications import NeedleCommandline

import torch
import torch.nn.functional as F


from torch_clustering import PyTorchGaussianMixture


def set_seed(seed: int = 42):
    """
    Set the random seed for common libraries and environments to ensure reproducible results.

    Args:
        seed (int): The random seed to set. Default is 42.
    """
    import os
    import random
    import numpy as np

    os.environ['PYTHONHASHSEED'] = str(seed)               # Fix Python's hash seed
    random.seed(seed)                                      # Fix built-in random generator
    np.random.seed(seed)                                   # Fix NumPy's random seed

    try:
        import torch
        torch.manual_seed(seed)                            # CPU random seed
        torch.cuda.manual_seed(seed)                       # Current GPU random seed
        torch.cuda.manual_seed_all(seed)                   # All GPUs random seed
        torch.backends.cudnn.deterministic = True          # Use deterministic algorithms
        torch.backends.cudnn.benchmark = False             # Disable auto-tuner
    except ImportError:
        pass  

    print(f"[INFO] Random seed set to {seed}")




def read_fastq_dedup_no_N(fastq_file):
    """
    Read a FASTQ file, remove duplicate sequences, and filter out sequences containing 'N'.

    Args:
        fastq_file (str): Path to the FASTQ file.

    Returns:
        list of str: Unique sequences without 'N'.
    """
    unique_sequences = set()  # To track unique sequences
    result = []               # Final sequence list

    with open(fastq_file, 'r') as f:
        while True:
            identifier = f.readline().strip()
            if not identifier:
                break  # End of file

            sequence = f.readline().strip()
            plus = f.readline().strip()
            quality = f.readline().strip()

            if 'N' in sequence:
                continue  # Skip sequences containing 'N'

            if sequence not in unique_sequences:
                unique_sequences.add(sequence)
                result.append(sequence)

    return result




def fastq_to_csv_with_counts(fastq_file, output_csv):
    """
    Read a FASTQ file, filter out sequences containing 'N', count the frequency of each sequence,
    and write the results to a CSV file sorted by frequency in descending order.

    Args:
        fastq_file (str): Path to the input FASTQ file.
        output_csv (str): Path to the output CSV file.

    Returns:
        tuple:
            list of tuples: [(sequence, count), ...] sorted by count in descending order.
            str: Path to the generated CSV file.
    """
    sequences = []

    with open(fastq_file, 'r') as f:
        while True:
            identifier = f.readline()
            if not identifier:
                break  # End of file
            seq = f.readline().strip()
            plus = f.readline()
            quality = f.readline()

            if 'N' in seq:
                continue  # Skip sequences containing 'N'

            sequences.append(seq)

    seq_counter = Counter(sequences)
    sorted_seqs = seq_counter.most_common()

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sequence', 'count'])
        for seq, count in sorted_seqs:
            writer.writerow([seq, count])

    return sorted_seqs, output_csv



import os
import torch
from tqdm import tqdm

def encode_sequences_to_embeddings(
    encoder_model,
    tokenizer,
    data_path_or_seqs: str,
    batch_size: int = 128,
    kmer_size: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Encode sequences into embedding vectors.

    Args:
        encoder_model: Pretrained encoder model.
        tokenizer: Tokenizer associated with the encoder model.
        data_path_or_seqs (str or list): Path to a text/FASTQ file or a list of sequences.
        batch_size (int): Batch size for processing. Default is 128.
        kmer_size (int): Size of k-mers used to reconstruct sequences. Default is 3.
        device (str): Device to run the model on ("cuda" or "cpu"). Default is CUDA if available.

    Returns:
        tuple:
            torch.Tensor: Tensor of shape (num_sequences, embedding_dim) containing all sequence embeddings.
            dict: Mapping from sequence string to its embedding tensor.
            dict: Mapping from flattened embedding tensor (as a tuple) to the corresponding sequence string.
    """

    encoder_model.to(device)

    # Handle input: either a list of sequences or a file path
    if not isinstance(data_path_or_seqs, list):
        if not os.path.isfile(data_path_or_seqs):
            raise ValueError(f"❌ Invalid file path: {data_path_or_seqs}")
        
        elif data_path_or_seqs.endswith(".fastq"):
            lines, _ = fastq_to_kmer_txt(data_path_or_seqs, k=kmer_size)
        
        else:  # Plain text file
            with open(data_path_or_seqs, "r") as f:
                lines = [kmers_sliding_windows(line.strip()) for line in f if line.strip()]
    else:
        lines = [kmers_sliding_windows(seq.strip()) for seq in data_path_or_seqs if seq.strip()]

    total_embeddings = []
    sequence_dict = {}
    tensor_to_sequence_dict = {}

    # Encode in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(lines), batch_size), desc="Encoding"):
            batch = tokenizer.batch_encode_plus(
                lines[i:i + batch_size],
                add_special_tokens=True,
                max_length=512,
                padding="longest",
                return_tensors="pt"
            ).to(device)

            input_ids = batch.data["input_ids"]
            outputs = encoder_model(input_ids)
            pooled_output = outputs  

            for idx in range(input_ids.size(0)):
                emb = pooled_output[idx].cpu().view(-1)
                total_embeddings.append(emb)

                tokens = tokenizer.convert_ids_to_tokens(input_ids[idx, 1:-1])
                decoded_seq = Three_mer2seq_with_check(tokens)

                sequence_dict[decoded_seq] = emb
                tensor_to_sequence_dict[tuple(emb.tolist())] = decoded_seq
      
    return torch.stack(total_embeddings), sequence_dict, tensor_to_sequence_dict



import os
import torch
import pandas as pd

def select_max_activity_by_cluster(
    seq_activity_dict: dict,
    encode_model,
    tokenizer,
    output_path: str = None,
    verbose: bool = True
) -> list:
    """
    Encode sequences using the given model, perform GMM clustering on the embeddings,
    and select the sequence with the highest activity from each cluster.

    Args:
        seq_activity_dict (dict): Dictionary mapping {sequence: activity}.
        encode_model: Model for generating sequence embeddings (should accept token IDs as input).
        tokenizer: DNA tokenizer (must implement `batch_encode_plus`).
        output_path (str, optional): Path to save the output CSV file. Default is None.
        verbose (bool): Whether to print selection results. Default is True.

    Returns:
        tuple:
            list: Sequences with the highest activity from each cluster.
            list: Corresponding activity values for the selected sequences.
    """
    device = next(encode_model.parameters()).device

    # Prepare sequences and activities
    seqs = list(seq_activity_dict.keys())
    activities = list(seq_activity_dict.values())

    # Convert sequences to k-mer tokenized form
    mers_sep_seq = [kmers_sliding_windows(seq) for seq in seqs]
    input_ids = tokenizer.batch_encode_plus(
        mers_sep_seq,
        add_special_tokens=True,
        max_length=512
    )["input_ids"]
    input_ids = torch.tensor(input_ids).to(device)

    # Encode sequences into embeddings
    with torch.no_grad():
        output = encode_model(input_ids)
        embeddings = output

    # Perform GMM clustering
    _, cluster_labels = GMM_embedding_clustering_pytorch(embeddings.cpu())

    # Create DataFrame with sequences, activity, and cluster labels
    df = pd.DataFrame({
        "DNA_sequence": seqs,
        "Activity_label": activities,
        "Cluster_label": cluster_labels
    })

    # Select the sequence with the highest activity per cluster
    result_df = df.loc[df.groupby("Cluster_label")["Activity_label"].idxmax()]

    # Save to CSV if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("Sequence,Activity\n")
            for _, row in result_df.iterrows():
                if verbose:
                    print(f"Cluster {row['Cluster_label']}: {row['DNA_sequence']} "
                          f"(Activity: {row['Activity_label']})")
                f.write(f"{row['DNA_sequence']},{row['Activity_label']}\n")

    if verbose and output_path:
        print(f"✅ Output file saved to: {output_path}")
    
    return result_df["DNA_sequence"].tolist(), result_df["Activity_label"].tolist()




def GMM_embedding_clustering_pytorch(sequence_embedding, k=10):
    """
    Perform Gaussian Mixture Model (GMM) clustering on sequence embeddings using PyTorch.

    Args:
        sequence_embedding (Iterable[torch.Tensor]): List or tensor of sequence embeddings.
        k (int): Number of clusters to fit. Default is 10.

    Returns:
        tuple:
            np.ndarray: Cluster center coordinates (shape: [k, embedding_dim]).
            np.ndarray: Predicted cluster labels for each sequence.
    """
    # Convert embeddings to a flat numpy list
    data = [seq.cpu().numpy().flatten() for seq in sequence_embedding]

    # Convert to a PyTorch tensor on GPU
    X = torch.tensor(data, dtype=torch.float32).cuda()

    # Initialize and fit GMM
    gmm = PyTorchGaussianMixture(n_clusters=k)
    pre_label = gmm.fit_predict(X)  # Shape: [n_samples, n_clusters]

    # Select the most probable cluster for each sequence
    pre_label = torch.argmax(pre_label.detach().clone(), dim=1)

    # Extract cluster centers
    centers = gmm.cluster_centers_

    # Move results to CPU and convert to numpy
    pre_label = pre_label.detach().cpu().numpy()
    centers = centers.detach().cpu().numpy()

    return centers, pre_label

def cluster_embeddings(
    sequence_embeddings,
    clustering_fn=GMM_embedding_clustering_pytorch
):
    """
    Cluster a list of embedding vectors using the specified clustering function.

    Args:
        sequence_embeddings (List[torch.Tensor]): List of embedding tensors.
        clustering_fn (Callable): Clustering function that takes embeddings and 
                                  returns a tuple (cluster_centers, labels).

    Returns:
        tuple:
            cluster_centers (np.ndarray or torch.Tensor): Coordinates of cluster centers.
            labels (np.ndarray): Cluster label for each sample.
    """
    cluster_centers, labels = clustering_fn(sequence_embeddings)
    return cluster_centers, labels

def get_cluster_center_sequence_embeddings(
    encoder_model,
    tokenizer,
    data_path: str,
    clustering_fn=GMM_embedding_clustering_pytorch,
    batch_size: int = 128,
    kmer_size: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    embeddings, seq_dict, tensor_to_seq = encode_sequences_to_embeddings(
        encoder_model, tokenizer, data_path, batch_size, kmer_size, device
    )
    cluster_centers, labels = cluster_embeddings(embeddings, clustering_fn)
    return cluster_centers, embeddings, seq_dict, tensor_to_seq

def get_cluster_center_seqs(data_dir,encode_model,decode_model, tokenizer,fastq_info=False,tSNE=False, model_down_dim=8):
    
    generated_seqs=[]
    
    cluster_center_emb, sequence_embedding, all_batch_sequence_dict,all_batch_tensor_seq_dict=get_cluster_center_sequence_embeddings(
        encoder_model=encode_model,
        tokenizer=tokenizer,
        data_path=data_dir,
    )
    
    if tSNE:
        tSNE_xy= t_SNE_sklearn(sequence_embedding)
        os.makedirs("tSNE_results", exist_ok=True)
        with open("tSNE_results/tSNE_xy.csv", "w") as f:
            f.write("Sequence,t-SNE_X,t-SNE_Y\n")
            for seq, coord in zip(all_batch_sequence_dict.keys(), tSNE_xy):
                f.write(f"{seq},{coord[0]},{coord[1]}\n")
        print("t-SNE coordinates saved to tSNE_results/tSNE_xy.csv")

    for point in tqdm(cluster_center_emb, desc=f"Cluster center decoding ... "):
        point_tensor = torch.tensor(point, dtype=torch.float32).to(next(encode_model.parameters()).device).reshape(1, -1, model_down_dim)
        with torch.no_grad():
            
            generated_seq = DNABERT_mask_seq_genernate_nolinkers(point_tensor, decode_model,  encode_model, tokenizer)
            if not generated_seq:
                continue
            
            core_seq = generated_seq
            generated_seqs.append(core_seq)

    return generated_seqs,sequence_embedding, all_batch_sequence_dict,all_batch_tensor_seq_dict


def t_SNE_sklearn(all_seq_embedding_list):
    """
    Reduce embedding vectors to 2 dimensions using t-SNE.

    Args:
        all_seq_embedding_list (list): List of input embedding vectors (torch.Tensor or list).

    Returns:
        np.ndarray: 2D embeddings after t-SNE dimensionality reduction.
    """
    data = []
    for seq in all_seq_embedding_list:
        # Convert torch.Tensor to numpy array if needed
        if isinstance(seq, list):
            data.append(seq)
        else:
            data.append(seq.cpu().numpy().flatten())
    
    data = np.array(data)

    tsne = TSNE(n_components=2, init='random', random_state=42)
    embedded_data = tsne.fit_transform(data)

    return embedded_data



def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    Args:
        seq (str): Input DNA sequence.

    Returns:
        str: Reverse complement of the input sequence.
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))



def DNA_score_compute(
    seq_list: List[str],
    motif_len: int = 8,
    score_file: Union[str, None] = None,
    verbose: bool = True
) -> List[float]:
    """
    Compute total motif match scores for a list of DNA sequences and return as a list.

    Args:
        seq_list (List[str]): List of DNA sequences.
        motif_len (int): Length of motifs to match, default is 8.
        score_file (str): Path to the file containing motifs and their scores.
        verbose (bool): Whether to print detailed information.

    Returns:
        List[float]: Total motif match scores for each sequence.
    """
    motif_score = {}

    # Load motif scoring table
    if score_file and os.path.isfile(score_file):
        with open(score_file, "r") as f:
            for line in f:
                if line.strip().startswith("ID"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                motif = parts[0]
                score = float(parts[-1])
                motif_score[motif] = score
                motif_score[reverse_complement(motif)] = score
    else:
        raise ValueError("❌ Invalid score file path.")

    # Compute scores for each sequence
    scores = []
    for seq in seq_list:
        total_score = 0.0
        matched_motifs = []

        for i in range(len(seq) - motif_len + 1):
            sub_motif = seq[i:i + motif_len]
            if sub_motif in motif_score:
                matched_motifs.append(sub_motif)
                total_score += motif_score[sub_motif]

        scores.append(total_score)
        if verbose:
            if matched_motifs:
                print(f"[INFO] Sequence: {seq} | Matched motifs: {len(matched_motifs)} | Total score: {total_score}")
            else:
                print(f"[INFO] Sequence: {seq} | No matching motifs found.")

    return scores


def kmers_sliding_windows(seq: str, kmer_size: int = 3) -> str:
    """
    Generate k-mer subsequences from the input DNA sequence using a sliding window.

    Args:
        seq (str): DNA sequence.
        kmer_size (int): Size of each k-mer (default is 3).

    Returns:
        str: A string of space-separated k-mers.
    """
    return " ".join(seq[i:i + kmer_size] for i in range(len(seq) - kmer_size + 1))



def find_nearest_neighbor(query_embedding, embeddings, k=100):
    """
    Find the k nearest neighbor embeddings to a query embedding, ensuring uniqueness.

    Args:
        query_embedding (torch.Tensor): The embedding to compare against.
        embeddings (torch.Tensor): The pool of embeddings to search.
        k (int): Number of nearest neighbors to return (default 100).

    Returns:
        torch.Tensor: Tensor of k unique nearest neighbor embeddings.
    """
    embeddings = embeddings.to(query_embedding.device)

    # Compute Euclidean distances between query and all embeddings
    distances = torch.norm(embeddings - query_embedding, dim=1)

    # Get initial top-k indices based on smallest distances
    values, indices = torch.topk(distances, k=k, largest=False)

    # Ensure uniqueness by increasing k if duplicates exist
    temp_k = k
    while len(set(tuple([tuple(i) for i in embeddings[indices].tolist()]))) < k:
        temp_k += 1
        values, indices = torch.topk(distances, k=temp_k, largest=False)

    tensor_group = embeddings[indices]
    unique_tensors = []

    # Filter out duplicates to keep only unique embeddings
    for idx in range(len(tensor_group)):
        selected_tensor = tensor_group[idx]
        if all(torch.equal(selected_tensor, t) is False for t in unique_tensors):
            unique_tensors.append(selected_tensor)

    return torch.stack(unique_tensors)




def fastq_to_kmer_txt(fastq_path: str, k: int = 3) -> Tuple[List[str], str]:
    """
    Convert sequences in a FASTQ file to k-mer formatted text.
    Each sequence is one line, with k-mers separated by spaces.
    The output file name is the original file name appended with '_mers_separated.txt'.

    Args:
        fastq_path (str): Path to the FASTQ file.
        k (int): k-mer size (default 3).

    Returns:
        Tuple[List[str], str]: A list of k-mer strings and the output .txt file path.
    """
    def generate_kmers(sequence: str, k: int) -> List[str]:
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    all_kmers = []
    base, ext = os.path.splitext(fastq_path)
    output_txt_path = f"{base}_mers_separated.txt"

    with open(fastq_path, "r") as f_in, open(output_txt_path, "w") as f_out:
        line_num = 0
        for line in f_in:
            line_num += 1
            if line_num % 4 == 2:  # FASTQ sequence lines appear every 4 lines
                seq = line.strip().upper()
                if "N" in seq:
                    print(f"⚠️ Warning: 'N' found on line {line_num}, skipping this sequence.")
                    continue
                kmers = generate_kmers(seq, k)
                kmers_str = " ".join(kmers)
                all_kmers.append(kmers_str)
                f_out.write(kmers_str + "\n")

    print(f"✅ Conversion completed. Output saved to: {output_txt_path}")
    return all_kmers, output_txt_path



def Three_mer2seq_with_check(kmer_list: list) -> str:
    if not kmer_list:
        return ""
    full_seq = kmer_list[0]
    for i in range(1, len(kmer_list)):
        prev = kmer_list[i - 1]
        curr = kmer_list[i]
        if prev[1:] != curr[:2]:
            raise ValueError(f"Inconsistent overlap between '{prev}' and '{curr}' at position {i}")
        full_seq += curr[-1]
    return full_seq

# def calculate_similarities_pairwise2(reference_sequences: List[str], generated_sequences: List[str]) -> List[Tuple[str, float]]:
#     results = []
#     for gen_seq in generated_sequences:
#         max_similarity = 0
#         for ref_seq in reference_sequences:
#             alignments = pairwise2.align.globalxx(ref_seq, gen_seq)
#             best_alignment = alignments[0]
#             aligned_seq1, aligned_seq2, score, begin, end = best_alignment
#             identity_count = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
#             identity_percentage = (identity_count / len(aligned_seq1)) * 100
#             max_similarity = max(max_similarity, identity_percentage)
#         results.append((gen_seq, max_similarity))
#     return results




def calculate_identity_needle(args) -> Tuple[str, float]:
    ref_seq, gen_seq, tag = args

    temp_seq1 = tempfile.NamedTemporaryFile(delete=False)
    temp_seq2 = tempfile.NamedTemporaryFile(delete=False)
    temp_out = tempfile.NamedTemporaryFile(delete=False)

    try:
        with open(temp_seq1.name, 'w') as f1, open(temp_seq2.name, 'w') as f2:
            f1.write(f">ref_{tag}\n{ref_seq}\n")
            f2.write(f">gen_{tag}\n{gen_seq}\n")

        needle_cline = NeedleCommandline(
            asequence=temp_seq1.name,
            bsequence=temp_seq2.name,
            gapopen=10,
            gapextend=0.5,
            outfile=temp_out.name,
            auto=True
        )
        needle_cline()

        with open(temp_out.name, 'r') as result_file:
            result = result_file.read()

        match = re.findall(r'# Identity:\s*\d+/\d+\s*\(\s*(\d+\.\d+)\s*%\)', result)
        identity = float(match[0]) if match else 0.0

        return (tag, identity)

    finally:
        os.remove(temp_seq1.name)
        os.remove(temp_seq2.name)
        os.remove(temp_out.name)


def calculate_similarities_needle(reference_sequences: List[str], generated_sequences: List[str]) -> List[Tuple[str, float]]:
    tasks = []
    for idx, gen_seq in enumerate(generated_sequences):
        for jdx, ref_seq in enumerate(reference_sequences):
            tag = f"{idx}_{jdx}"
            tasks.append((ref_seq, gen_seq, tag))

    results = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(calculate_identity_needle, tasks), total=len(tasks), desc="Processing"):
            results.append(result)

    # 聚合每个 generated_sequence 的最大 identity
    id_map = {}
    for tag, identity in results:
        gen_idx = int(tag.split("_")[0])
        if gen_idx not in id_map:
            id_map[gen_idx] = identity
        else:
            id_map[gen_idx] = max(id_map[gen_idx], identity)

    return [(generated_sequences[i], id_map.get(i, 0.0)) for i in range(len(generated_sequences))]


def load_partial_model(model, checkpoint_path):
    """
    Load partial model weights from a checkpoint file into the provided model.

    Args:
        model (torch.nn.Module): The model into which the weights will be loaded.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: The model with loaded weights.
        dict: The checkpoint's state dictionary (optional, if you want to restore other information like optimizer state).
    """
    def is_subsequence(s, t):
        """判断 s 是否是 t 的子序列"""
        it = iter(t)
        return all(c in it for c in s)

    # Load the checkpoint
    checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')



    # Get the model's current state dictionary
    model_state_dict = model.state_dict()

    # Update only the layers that exist in both the model and checkpoint
    for name, param in checkpoint_state_dict.items():
        matched_key = None
        for model_key in model_state_dict.keys():
            if (name in model_key or model_key in name) and model_state_dict[model_key].shape == param.shape:
                matched_key = model_key
                break

        if matched_key:
            print(f"Layer {matched_key} (matched from {name}) weight loaded!")
            model_state_dict[matched_key].copy_(param)
        else:
            print(f"Skipping {name}, no matching key found (or shape mismatch).")

    # Load the updated state dictionary back into the model
    model.load_state_dict(model_state_dict)

    return model, checkpoint_state_dict


def load_latest_checkpoint(model, checkpoint_dir):
    """
    Load the latest checkpoint based on the highest number in the filename.

    Args:
        model (torch.nn.Module): The model into which the weights will be loaded.
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        torch.nn.Module: The model with loaded weights.
        dict: The checkpoint's state dictionary (optional, if you want to restore other information like optimizer state).
    """
    # List all checkpoint files in the directory
    if os.path.isfile(checkpoint_dir):
        checkpoint_path=checkpoint_dir
    else:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint') ]
         # Extract the numbers from filenames and find the one with the highest number
        checkpoint_numbers = [int(f.split('-')[1]) for f in checkpoint_files]
        latest_checkpoint_number = max(checkpoint_numbers)
        latest_checkpoint_file = f"checkpoint-{latest_checkpoint_number}"
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file,"pytorch_model_trainable.bin")
        print(f"Loading checkpoint: {checkpoint_path}")

        if not checkpoint_files:
            print("No checkpoint files found in the directory.")
            return model, None

   

    # Load the checkpoint
    model, checkpoint = load_partial_model(model, checkpoint_path)

    return model, checkpoint

def top_k_top_p_sampling(logits, top_k=5, top_p=0.9, temperature=1.2):
    """
    Performs Top-K and Top-P (nucleus) sampling on logits with temperature scaling.

    Args:
        logits (Tensor): Input logits tensor of shape (vocab_size,).
        top_k (int): Keep only the top_k highest probability logits. Default is 5.
        top_p (float): Keep the smallest set of logits whose cumulative probability >= top_p. Default is 0.9.
        temperature (float): Temperature value for scaling logits. Higher values increase randomness. Default is 1.2.

    Returns:
        Tensor: Index of the sampled token (shape: [1, 1]).
    """
    logits = logits.squeeze()
    logits = logits / temperature
    topk_logits, topk_indices = torch.topk(logits, top_k)
    topk_probs = F.softmax(topk_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(topk_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_index_in_sorted = torch.multinomial(sorted_probs, 1)
    sampled_index_in_topk = sorted_indices.gather(-1, sampled_index_in_sorted)
    sampled_index = topk_indices.gather(-1, sampled_index_in_topk)
    
    return sampled_index.view(1, -1)



def generate_3mers():
    return [''.join(p) for p in product('ACGT', repeat=3)]

def hamming(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))


def repair(tokens):
    vocab = generate_3mers()
    n = len(tokens)
    dp = [defaultdict(lambda: (float('inf'), None)) for _ in range(n)]
    dp[0][tokens[0]] = (0, None)

    for i in range(1, n):
        for prev_token in dp[i - 1]:
            prev_cost, _ = dp[i - 1][prev_token]
            expected_prefix = prev_token[1:]

            candidates = [t for t in vocab if t[:2] == expected_prefix]
            for cand in candidates:
                cost = hamming(cand, tokens[i])
                total_cost = prev_cost + cost
                if total_cost < dp[i][cand][0]:
                    dp[i][cand] = (total_cost, prev_token)
                    
    last_token = min(dp[-1], key=lambda t: dp[-1][t][0])
    repaired = [last_token]

    for i in range(n - 1, 0, -1):
        last_token = dp[i][last_token][1]
        repaired.append(last_token)

    return repaired[::-1]



def DNABERT_mask_seq_genernate_nolinkers(input_embedding, decode_model, encode_model, tokenizer ,lowest_probs_mask_ratio=0.15,
                       pre_link=None, rever_link=None, iterative_optimize=3, top_k=20, top_p=1, temperature=1):
    logits = decode_model(input_embedding)

    once_mask_token_probability = torch.softmax(logits, dim=2)
    logits[:, 1:-1, tokenizer.all_special_ids] = -1e-6
    once_predict_seq_id = top_k_top_p_sampling(logits, top_k, top_p, temperature)
    


    decoded_tokens = tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    repaired_tokens=repair(decoded_tokens)
    encoded = torch.tensor(tokenizer.encode(" ".join(repaired_tokens)), dtype=torch.long, device=once_predict_seq_id.device)
    once_predict_seq_id= encoded.unsqueeze(dim=0)

    """mask generate optimization"""

    mask_num = int(lowest_probs_mask_ratio * len(input_embedding[0]))

    topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
    for i in range(iterative_optimize):

        min_values, min_indices = torch.topk(topk_values.view(-1), k=mask_num, largest=False)
        min_indices_list = min_indices.tolist()
        temp_list=copy.deepcopy(min_indices_list)
        for idx in temp_list:
            min_indices_list.extend([idx-1,idx+1])
        min_indices_list=list(set(min_indices_list))
        min_indices_list = [item for item in min_indices_list if item > -1 and item <once_predict_seq_id.size()[1]]

        once_predict_seq_id[0, min_indices_list] = tokenizer.mask_token_id

        mask_model_predict_logits = encode_model(once_predict_seq_id,predict_token=True)
        mask_model_predict_logits[:, 1:-1, tokenizer.all_special_ids] = -1e-6
        iterative_mask_token_probability = torch.softmax(mask_model_predict_logits, dim=2)
        once_mask_token_probability[:, min_indices, :] = iterative_mask_token_probability[:, min_indices, :]
        topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)

        once_predict_seq_id[0, min_indices_list] = torch.argmax(mask_model_predict_logits, dim=2)[0, min_indices_list]
        once_predict_seq_id[0, 0] = tokenizer.cls_token_id
        once_predict_seq_id[0, -1] = tokenizer.sep_token_id

    
    decoded_tokens= tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    repaired_tokens=repair(decoded_tokens)
    decoded_text = Three_mer2seq_with_check(repaired_tokens)
    once_gennerated_sequence = decoded_text
    
    if pre_link and rever_link:
        final_gennerated_sequence = pre_link+once_gennerated_sequence+rever_link
    else:
        final_gennerated_sequence=once_gennerated_sequence

    return final_gennerated_sequence





def nt_mask_seq_genernate_nolinkers(input_embedding, decode_model,  encode_model, tokenizer ,pre_link, rever_link, lowest_probs_mask_ratio=0.15,
                       iterative_optimize=3, top_k=10, top_p=0.8, temperature=1,):
    logits = decode_model(input_embedding)
    once_mask_token_probability = torch.softmax(logits, dim=2)
    # once_predict_seq_id = torch.argmax(once_mask_token_probability , dim=2)
    logits[:, :, tokenizer.all_special_ids] = -100
    once_predict_seq_id = top_k_top_p_sampling(logits, top_k, top_p, temperature)


    once_predict_seq_id[0, 0] = tokenizer.cls_token_id


    decoded_text = tokenizer.decode(once_predict_seq_id[0,1:].view(-1))
    once_gennerated_sequence = decoded_text.replace(" ","")

    topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
    """mask generate optimization"""

    mask_num = max(1,int(lowest_probs_mask_ratio * len(input_embedding[0])))
    # mask_num = 1

    for idx, m in enumerate(once_predict_seq_id[0, :].tolist()[1:]):
        if m in tokenizer.all_special_ids:
            once_predict_seq_id[0, idx + 1] = tokenizer.mask_token_id
            
    mask_model_predict_probs=encode_model(once_predict_seq_id,predict_token=True)
    mask_model_predict_probs[:, :, tokenizer.all_special_ids] = -100
    mask_model_predict_indices= torch.argmax(mask_model_predict_probs,dim=2)
    mask_indices=torch.where(once_predict_seq_id == tokenizer.mask_token_id)[1]
    once_predict_seq_id[:,mask_indices]=mask_model_predict_indices[:,mask_indices]
    once_mask_token_probability[:, mask_indices, :] = torch.softmax(mask_model_predict_probs,dim=2)[:, mask_indices, :]
    decoded_text = tokenizer.decode(once_predict_seq_id[0,1:].view(-1))
    once_gennerated_sequence = decoded_text.replace(" ","")
    


    for i in range(iterative_optimize):


        min_values, min_indices = torch.topk(topk_values.view(-1), k=mask_num, largest=False)
        min_indices_list = min_indices.tolist()
        temp_list=copy.deepcopy(min_indices_list)

        min_indices_list=list(set(min_indices_list))
        min_indices_list = [item for item in min_indices_list if item > -1 and item <once_predict_seq_id.size()[1]]

        once_predict_seq_id[0, min_indices_list] = tokenizer.mask_token_id


        mask_model_predict_logits = encode_model(once_predict_seq_id,predict_token=True)
        iterative_mask_token_probability = torch.softmax(mask_model_predict_logits, dim=2)
        once_mask_token_probability[:, min_indices, :] = iterative_mask_token_probability[:, min_indices, :]

        topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
        mask_model_predict_probs[:, :,tokenizer.all_special_ids] = -100
        once_predict_seq_id[0, min_indices_list] = torch.argmax(mask_model_predict_probs, dim=2)[0, min_indices_list]
        once_predict_seq_id[0, 0] = tokenizer.cls_token_id
    

    
    decoded_text = tokenizer.decode(once_predict_seq_id[0,1:].view(-1))
    once_gennerated_sequence = decoded_text.replace(" ","")
    

    if pre_link and rever_link:
        final_gennerated_sequence = pre_link+once_gennerated_sequence+rever_link
    else:
        final_gennerated_sequence=once_gennerated_sequence

    return final_gennerated_sequence





def mask_seq_generate_nolinkers_evo(
    input_embedding,
    decode_model,
    encoder,
    tokenizer,
    pre_link,
    rever_link,
    top_k=2,
    top_p=1,
    temperature=1,
):
    """
    Sequence generation with no linkers using top-k/top-p sampling.

    Args:
        input_embedding (Tensor): Input embedding tensor.
        decode_model (nn.Module): Decoder model to generate logits.
        tokenizer: Tokenizer object with detokenize method.
        pre_link (str): Prefix sequence to attach.
        rever_link (str): Suffix sequence to attach.
        top_k (int, optional): Top-k sampling parameter. Defaults to 3.
        top_p (float, optional): Top-p (nucleus) sampling parameter. Defaults to 0.8.
        temperature (float, optional): Sampling temperature. Defaults to 1.

    Returns:
        str: Final generated sequence.
    """
    ACTG_ids=tokenizer.tokenize("ACTG")
    ACTG_ids=[int(i) for i in ACTG_ids]
    def mask_logits_to_subspace(logits, allowed_token_ids):
        """
        Mask logits so that only tokens in allowed_token_ids are valid for sampling.
        
        Args:
            logits (Tensor): shape [batch, seq_len, vocab_size]
            allowed_token_ids (list[int]): list of allowed token ids
        
        Returns:
            masked_logits (Tensor): logits with only allowed token_ids unmasked
        """
        mask = torch.full_like(logits, float('-inf'))  
        allowed_token_ids_tensor = torch.tensor(allowed_token_ids, device=logits.device)


        mask[:, :, allowed_token_ids_tensor] = logits[:, :, allowed_token_ids_tensor]
        return mask

    logits = decode_model(input_embedding)
    masked_logits =mask_logits_to_subspace(logits,ACTG_ids)
    # Top-k, top-p, temperature sampling to get predicted token ids
    once_predict_seq_id = top_k_top_p_sampling(masked_logits[:,:-1,:], top_k, top_p, temperature)

    # Detokenize to get sequence string
    once_generated_sequence = tokenizer.detokenize(once_predict_seq_id.tolist()[0])

    # Optionally add pre-link and rever-link
    if pre_link and rever_link:
        final_generated_sequence = pre_link + once_generated_sequence + rever_link
    else:
        final_generated_sequence = once_generated_sequence

    return final_generated_sequence



