from InstructNA_frameworks.utills import fastq_to_csv_with_counts
import pandas as pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract top 10 frequent sequences from FASTQ and save to CSV")
    parser.add_argument("--fastq_file", type=str, default=None, help="Path to the input FASTQ file")
    parser.add_argument("--output_csv", type=str, default="./top10.csv", help="Path to the output CSV file")

    args = parser.parse_args()
    seq_counter,_ = fastq_to_csv_with_counts(args.fastq_file, args.output_csv)  
    top10 = seq_counter[:10]
    with open(args.output_csv, 'w', newline='') as csvfile:
        for seq,counter in seq_counter[:10]:           
            csvfile.write(f"{seq}\n")
    print(f"Top 10 sequences saved to {args.output_csv}")