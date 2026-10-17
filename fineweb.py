"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard

# Create the cache directory locally if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 1. Download the dataset (sample-10BT is ~10 billion tokens)
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# 2. Initialize the gpt-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # 50256

def tokenize(doc):
    # Tokenizes a single document and returns a uint16 numpy array
    tokens = [eot]  # Delimit documents with <|endoftext|>
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token ID too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__ == "__main__":
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold the current shard (100M tokens)
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # Check if current shard has enough space
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Fill the remaining space in the current shard and save it
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_train_{shard_index:06d}")
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                
                # Carry over leftover tokens to the new shard
                leftover = tokens[remainder:]
                all_tokens_np[0:len(leftover)] = leftover
                token_count = len(leftover)

        # Write any trailing leftover tokens to a final shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_train_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])