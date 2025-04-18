import os, sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F
from gpt2 import estimate_loss, BigramLanguageModel
torch.manual_seed(522170)

# check cuda availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize a seed for reproducibility
batch_size = 64     # batch size
block_size = 256     # maximum context length for prediction
iterations = 5000  # number of iterations for training
eval_interval = 500 # print per X iterations
eval_iters = 200    # num iterations used for evaluation (to get an average)
max_tokens = 1000    # maximum number of tokens to generate
n_embed    = 384     # embedding dimension
nhead      = 6      # number of heads for MHA
n_layers   = 5      # number of blocks/decoder layers to have
learning_rate = 3e-4 # learning rate
dropout    = 0.2    # dropout rate

# 1. model data dir
indir = sys.argv[1]
if not os.path.isdir(indir):
    raise FileNotFoundError(f"Directory {indir} not found!")

## read input from tinyshakespeare.txt
#with open('input.txt', 'r') as f:
#    text = f.read().strip()

## obtain the list of unique characters
#chars = sorted(list(set(text)))
#vocab_size = len(chars)
#
## construct some mapping between integers and characters
## tokenization in a sense
#stoi = {ch: i for i, ch in enumerate(chars)}
#itos = {i: ch for i, ch in enumerate(chars)}

# read pretrained tokenizer from local
vocab_size = 10000
tokenizer_path = os.path.join(indir, 'tokenizer.json')
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    raise FileNotFoundError(f"tokenizer not found at: {tokenizer_path}")
model = BigramLanguageModel(vocab_size)

# try to find the input model
input_model = os.path.join(indir, 'gpt2.model')
# use the latest version if any
idx = 1
while os.path.exists(input_model + f'.{idx}'):
    idx += 1
if os.path.exists(input_model + f'.{idx}'):
    input_model = input_model + f'.{idx}'
if not os.path.exists(input_model):
    print(f"Model path {input_model} does not exist!")
    exit(1)
model.load_state_dict(torch.load(input_model, weights_only=True))
summary(model)

m = model.to(device)
model.eval()

## print out the final training loss and validation loss
#encoded = tokenizer.encode(text)
#data = torch.tensor(encoded.ids, dtype=torch.long)
#n = int(0.9*len(data))
#train_data = data[:n]
#val_data = data[n:]
#losses = estimate_loss(model, train_data, val_data, batch_size,
#        block_size, eval_iters)
#print(f"pretrained model: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
for nxt_token in model.generate(context, max_tokens=max_tokens):
    print(tokenizer.decode(nxt_token[0].tolist(), skip_special_tokens=True), end='')
print('')
#generated = tokenizer.decode(
#    model.generate(context, max_tokens=max_tokens)[0].tolist(),
#    skip_special_tokens=True,
#    )
#print(generated)
