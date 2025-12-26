# Activity Seed Construction in InstructNA

InstructNA constructs **activity seeds** to guide sequence generation when activity-labeled sequences are limited or unavailable.  
The seed construction strategy integrates **sequence enrichment**, **latent-space clustering**, and **activity-aware expansion**.

---

## Seed Construction Steps

### 1. **Frequent Sequence Selection**
High-frequency sequences are first extracted from SELEX or other high-throughput sequencing data.  
These enriched sequences are more likely to be functionally relevant and serve as initial seed candidates.

### 2. **Latent-Space Clustering and Generation**
All SELEX sequences are encoded into the InstructNA latent space and clustered using a **Gaussian Mixture Model (GMM)**.  
Representative latent centers are then decoded to generate additional candidate seed sequences.

### 3. **Activity-Guided Seed Expansion**
After obtaining activity measurements for sequences from Steps 1 and 2, sequences in the SELEX dataset that are closest (in latent space) to high-activity representations are selected to further expand the seed set.

---

The resulting activity seeds jointly capture **sequence diversity** and **functional relevance**, providing reliable starting points for downstream activity-guided sequence generation.

---

## Usage

All scripts in this directory are implemented using **Hydra configuration**.  
You can customize the variables in `../conf/seeds_construct` to control the behavior of the scripts.

---

### 1. **Frequent Sequence Selection** & **Latent-Space Clustering and Generation**

To run this step, you need:
- A **SELEX FASTQ file**, and  
- A **trained InstructNA model checkpoint** trained on the corresponding SELEX dataset.

Run the following command:

```bash
python 1_get_top_and_GMM_seqs.py \
  base.data.fastq_dir=/PATH/to/SELEX_FASTQ_FILE \
  base.model.InstructNA_model_path=/PATH/to/InstructNA_MODEL_CKPT
```

This script performs:
- **Frequent Sequence Selection**, and  
- **Latent-Space Clustering and Sequence Generation (GMM-based)**

and outputs the corresponding sequence sets for downstream analysis.

---

### 2. **Activity-Guided Seed Expansion (Optional)**

After measuring the activity of the sequences obtained in **Step 1**, prepare a CSV file with the following format:

```csv
seq,act
ACCCTAATTATATTAATTAG,24
AATTAGCGACTAATTACAAA,125
```

Specify the CSV file path via Hydra when running the script:

```bash
python 2_get_near_seqs.py \
  base.data.seq_acts=/PATH/to/constructed_csv_file.csv \
  base.data.fastq_dir=/PATH/to/SELEX_FASTQ_FILE \
  base.model.InstructNA_model_path=/PATH/to/InstructNA_MODEL_CKPT
```

This step expands the initial seed set by identifying sequences in the SELEX dataset that are close (in latent space) to high-activity seeds.  
The output file **`near_seqs_in_SELEX.csv`** contains candidate sequences for subsequent activity measurement.

---

### 3. **Final k-Seed Selection**

After obtaining activity measurements for all candidate sequences, prepare a CSV file containing all sequences and their activities in the following format:

```csv
seq,act
ACCCTAATTATATTAATTAG,24
AATTAGCGACTAATTACAAA,125
```

Then run the final seed selection script:

```bash
python 3_final_seed_selection.py \
  base.data.seq_acts=/PATH/to/constructed_csv_file.csv \
  base.model.InstructNA_model_path=/PATH/to/InstructNA_MODEL_CKPT \
  selection_k=k # the selected sequences number
```

This step selects the diverse top-*k* activity seeds and generates **`Final_seeds.csv`**, which can be directly used for **HC-HEBO-based sequence generation**.
