from InstructNA_frameworks.utills import fastq_to_csv_with_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert FASTQ to frequency CSV")
    parser.add_argument("--fastq_file", type=str, required=True, help="Path to the input FASTQ file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    
    args = parser.parse_args()
    # Read FASTQ file and convert to CSV
    fastq_to_csv_with_counts(args.fastq_file, args.output_csv)