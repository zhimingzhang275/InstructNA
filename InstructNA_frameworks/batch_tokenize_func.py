from InstructNA_frameworks.utills import kmers_sliding_windows

def evo_tokenize_batch(seqs:list,tokenizer):
    return [tokenizer.tokenize(seq)+[tokenizer.eos_id] for seq in seqs]

def NT_tokenize_batch(Sequence,Tokenizer, ):

    tokens_ids = Tokenizer.batch_encode_plus(Sequence,  padding="longest")["input_ids"]
    return tokens_ids


def DNABERT_3mer_tokenize_batch(Sequence,Tokenizer, add_special_tokens=True, max_length=512):
    mers_sep_seq = [kmers_sliding_windows(seq) for seq in Sequence]
    tokens_ids = Tokenizer.batch_encode_plus(mers_sep_seq, add_special_tokens=add_special_tokens, max_length=max_length)["input_ids"]
    return tokens_ids