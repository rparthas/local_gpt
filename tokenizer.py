import torch

text = "This is some text dataset hello, and hi some words!"
# get the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

text = "안녕하세요 👋 hello world 🤗"
print([ord(x) for x in text])

# UTF-8 encoding
utf8_bytes = list(text.encode("utf-8"))
print(f"UTF-8: {utf8_bytes}")

# UTF-16 encoding
utf16_bytes = list(text.encode("utf-16"))
print(f"UTF-16: {utf16_bytes}")

# UTF-32 encoding
utf32_bytes = list(text.encode("utf-32"))
print(f"UTF-32: {utf32_bytes}")

text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to "support Unicode" in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception."""

print(f"Text: {text}")
print(f"Length in characters: {len(text)}")

# Step 2: Encode the text to UTF-8 bytes and convert to list of integers
tokens = list(text.encode("utf-8"))
print(f"UTF-8 encoded bytes: {tokens[:50]}...")  # Show first 50 bytes
print(f"Length in bytes: {len(tokens)}")


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# Step 3: Find the most common consecutive pair using get_stats
stats = get_stats(tokens)
print(f"Total number of unique pairs: {len(stats)}")

# Show top 10 most frequent pairs
top_pairs = sorted([(count, pair) for pair, count in stats.items()], reverse=True)[:10]
print("\nTop 10 most frequent pairs:")
for count, pair in top_pairs:
    print(f"  {pair}: {count} times")

# Step 4: Get the most frequent pair using max() function
most_frequent_pair = max(stats, key=stats.get)
print(f"Most frequent pair: {most_frequent_pair}")
print(f"Occurs {stats[most_frequent_pair]} times")

# Convert bytes back to characters to see what this pair represents
char1 = chr(most_frequent_pair[0])
char2 = chr(most_frequent_pair[1])
print(f"This represents: '{char1}' + '{char2}'")

# Step 5: Prepare to merge - create new token ID
# Current tokens are 0-255 (256 possible values), so new token will be 256
new_token_id = 256
print(f"Will replace pair {most_frequent_pair} with new token ID: {new_token_id}")
print(f"Ready to implement merge function...")


# Step 6: Implement the merge function
def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2  # skip over the pair
        else:
            newids.append(ids[i])
            i += 1
    return newids


# Step 7: Apply merge to our actual tokens
# Merge the most frequent pair (101, 32) with token ID 256
tokens2 = merge(tokens, most_frequent_pair, new_token_id)

print(f"Original length: {len(tokens)}")
print(f"After merge length: {len(tokens2)}")
print(f"Reduction: {len(tokens) - len(tokens2)} tokens")

# Verify the merge worked
print(f"\nOccurrences of new token {new_token_id}: {tokens2.count(new_token_id)}")
print(
    f"Occurrences of old pair in original: {sum(1 for i in range(len(tokens) - 1) if (tokens[i], tokens[i + 1]) == most_frequent_pair)}"
)

# Verify old pair is gone
old_pair_count = sum(
    1
    for i in range(len(tokens2) - 1)
    if (tokens2[i], tokens2[i + 1]) == most_frequent_pair
)
print(f"Occurrences of old pair in new tokens: {old_pair_count}")

# Step 8: Iterate the BPE algorithm
# Now we repeat: find most common pair, merge it, repeat...
# Let's do a few more iterations

current_tokens = tokens2
vocab_size = 257  # Started with 256, now have 257

print("BPE Training Progress:")
print(f"Step 0: {len(tokens)} tokens, vocab size: 256")
print(f"Step 1: {len(current_tokens)} tokens, vocab size: {vocab_size}")

# Do a few more iterations
for step in range(2, 6):  # Steps 2-5
    # Find most common pair
    stats = get_stats(current_tokens)
    if not stats:  # No more pairs to merge
        break

    most_frequent_pair = max(stats, key=stats.get)

    # Merge it
    current_tokens = merge(current_tokens, most_frequent_pair, vocab_size)

    print(f"Step {step}: {len(current_tokens)} tokens, vocab size: {vocab_size + 1}")
    print(f"  Merged pair: {most_frequent_pair} -> {vocab_size}")

    vocab_size += 1

print(f"\nFinal: {len(current_tokens)} tokens, vocab size: {vocab_size}")

text = """
Having understood the BPE algorithm conceptually, we can now build the complete tokenizer with training, encoding,
and decoding functions. To get more representative statistics for byte pairs and produce sensible results,
we’ll use the entire blog post as our training text rather than just the first paragraph.
The raw text is encoded into bytes using UTF-8 encoding, then converted into a list of integers in Python for easier manipulation.
"""
tokens = list(text.encode("utf-8"))
print(f"UTF-8 encoded bytes: {tokens[:50]}...")  # Show first 50 bytes
print(f"Length in bytes: {len(tokens)}")

# BPE training
vocab_size = 276  # hyperparameter: the desired final vocabulary size
num_merges = vocab_size - 256
tokens = list(text.encode("utf-8"))

for i in range(num_merges):
    # count up all the pairs
    stats = get_stats(tokens)
    # find the pair with the highest count
    pair = max(stats, key=stats.get)
    # mint a new token: assign it the next available id
    idx = 256 + i
    # replace all occurrences of pair in tokens with idx
    tokens = merge(tokens, pair, idx)
    # print progress
    print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({stats[pair]} occurrences)")

merges = {
    (101, 32): 256,  # 'e' + ' '
    (100, 32): 257,  # 'd' + ' '
    (116, 101): 258,  # 't' + 'e'
    (115, 32): 259,  # 's' + ' '
    (105, 110): 260,  # 'i' + 'n'
}
# given ids (list of integers), return Python string
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


def decode(ids):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


print(decode([97]))
print(decode([128]))


def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while True:
        stats = get_stats(tokens)
        if len(tokens) < 2:
            break  # nothing to merge
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


print(encode("hello world"))
print(decode(encode("hello world")))
