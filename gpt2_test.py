#!/usr/bin/env python3
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
max_tokens = 50    # maximum number of tokens to generate
n_embed    = 384     # embedding dimension
nhead      = 6      # number of heads for MHA
n_layers   = 5      # number of blocks/decoder layers to have
learning_rate = 3e-4 # learning rate
dropout    = 0.2    # dropout rate

# 1. model data dir
indir = sys.argv[1]
if not os.path.isdir(indir):
    raise FileNotFoundError(f"Directory {indir} not found!")

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

# read in input
context = None
while True:
    line = input('\n>>> Write your input: ')
    if not line:
        break
    in_tensor = torch.tensor(tokenizer.encode(line).ids,
            dtype=torch.long, device=device).unsqueeze(0)
    if context == None:
        context = in_tensor
        #context = torch.tensor(tokenizer.encode(line).ids, dtype=torch.long,
        #        device=device).unsqueeze(0)
    else:
        new_context = torch.cat((context, in_tensor), dim=1)
        # last max_len item for context
        max_len = min(context.shape[1], block_size)
        context = new_context[:, -max_len:].clone()
    output = []
    for item in model.generate(context, max_tokens=max_tokens):
        nxt_token = item[0].tolist()
        output.append(nxt_token[0])
        print(tokenizer.decode(nxt_token, skip_special_tokens=True), end='')
    print('')
    out_tensor = torch.tensor(output, device=device, dtype=torch.long).unsqueeze(0)
    #print(context.shape, out_tensor.shape)
    context = torch.cat((context, out_tensor), dim=1)
    # update context 
