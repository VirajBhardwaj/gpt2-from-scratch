import os
import json
import regex as re

# ========================================== #
# 1. Basic Character/Byte Encoding Exploration
# ========================================== #

korean_text = "안녕하세요 👋 (hello in Korean!)"
print("Code points:", [ord(x) for x in korean_text])
print("UTF-8 Bytes:", list(korean_text.encode("utf-8")))

# ========================================== #
# 2. Simple Byte Pair Encoding (BPE) Helpers
# ========================================== #

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# ========================================== #
# 3. Simple BPE Training on Sample Text
# ========================================== #

text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 
The very name strikes fear and awe into the hearts of programmers worldwide.
We all know we ought to “support Unicode” in our software..."""

tokens = list(map(int, text.encode("utf-8")))

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:
        break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print(f"Original tokens length: {len(tokens)}")
print(f"Compressed ids length: {len(ids)}")
print(f"Compression ratio: {len(tokens) / len(ids):.2f}X")

# ========================================== #
# 4. Decoding and Encoding Implementations
# ========================================== #

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

test_string = "hello world"
encoded = encode(test_string)
decoded = decode(encoded)
print(f"Test encode/decode: '{test_string}' -> {encoded} -> '{decoded}'")

# ========================================== #
# 5. GPT Series Regex Splits
# ========================================== #

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print("Regex splits:", re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))

# ========================================== #
# 6. SentencePiece Simulation Setup
# ========================================== #
try:
    import sentencepiece as spm
    with open("toy.txt", "w", encoding="utf-8") as f:
        f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based systems.")
    
    options = dict(
        input="toy.txt",
        input_format="text",
        model_prefix="tok400",
        model_type="bpe",
        vocab_size=110,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        character_coverage=0.99995,
        byte_fallback=True,
        unk_id=0, 
        bos_id=1, 
        eos_id=2,
        pad_id=-1
    )
    spm.SentencePieceTrainer.train(**options)
    print("SentencePiece model trained successfully!")
except Exception as e:
    print("SentencePiece setup skipped or failed:", e)
