from InstructNA_frameworks.utills import read_fastq_dedup_no_N

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert FASTQ to unique sequences")
    parser.add_argument("--fastq_file", type=str, required=True, help="Path to the input FASTQ file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    
    args = parser.parse_args()
    # Read FASTQ file and convert to CSV
    seqs=read_fastq_dedup_no_N(args.fastq_file)
    with open(args.output_csv, 'w') as f:
        for seq in seqs:
            f.write(f"{seq}\n")