import os, sys
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from train_model import *
torch.manual_seed(522170)

# check cuda availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize a seed for reproducibility
batch_size = 32     # batch size
block_size = 16     # maximum context length for prediction
iterations = 3000  # number of iterations for training
eval_interval = 200 # print per X iterations
eval_iters = 200    # num iterations used for evaluation (to get an average)
max_tokens = 300    # maximum number of tokens to generate
n_embed    = 32     # embedding dimension

indir = sys.argv[1]
if not os.path.isdir(indir):
    raise FileNotFoundError(f"Directory {indir} not found!")

## read input from tinyshakespeare.txt
#input_path = os.path.join(indir, 'input.txt')
#with open(input_path, 'r') as f:
#    text = f.read().strip()

## obtain the list of unique characters
#chars = sorted(list(set(text)))
#vocab_size = len(chars)
#
## construct some mapping between integers and characters
## tokenization in a sense
#stoi = {ch: i for i, ch in enumerate(chars)}
#itos = {i: ch for i, ch in enumerate(chars)}

## encoding function to encode a string of characters to a list of integers
## decoding function to decode a list of integers back to a string of chars
#encode = lambda s: [stoi[c] for c in s]
#decode = lambda l: ''.join([itos[i] for i in l])

# read pretrained tokenizer from local
vocab_size = 10000
tokenizer_path = os.path.join(indir, 'tokenizer.json')
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    raise FileNotFoundError(f"tokenizer not found at: {tokenizer_path}")

model = BigramLanguageModel(vocab_size, n_embed, block_size)
model_path = os.path.join(indir, 'train.model')
model.load_state_dict(torch.load(model_path, weights_only=True))
m = model.to(device)
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
for nxt_token in model.generate(context, max_tokens=max_tokens):
    print(tokenizer.decode(nxt_token[0].tolist(), skip_special_tokens=True), end='')
print('')
