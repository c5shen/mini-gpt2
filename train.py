import os, sys
from tokenizers import Tokenizer, models, trainers
import torch
import torch.nn as nn
from torch.nn import functional as F
from train_model import *
torch.manual_seed(522170)

# check cuda availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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

# read input from tinyshakespeare.txt
input_path = os.path.join(indir, 'input.txt')
with open(input_path, 'r') as f:
    text = f.read().strip()

# obtain the list of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# construct some mapping between integers and characters
# tokenization in a sense
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

## encoding function to encode a string of characters to a list of integers
## decoding function to decode a list of integers back to a string of chars
#encode = lambda s: [stoi[c] for c in s]
#decode = lambda l: ''.join([itos[i] for i in l])
#data = torch.tensor(encode(text), dtype=torch.long)
#print(encode('You are brilliant'))
#print(decode(encode('You are brilliant')))

# use BPE to train on the tiny shakespeare dataset
vocab_size = 10000
tokenizer_path = os.path.join(indir, 'tokenizer.json')
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=vocab_size)
    tokenizer.train([input_path], trainer)
    # write to local
    tokenizer.save(tokenizer_path)

# convert tokenized data to tensor data
encoded = tokenizer.encode(text)
data = torch.tensor(encoded.ids, dtype=torch.long)
#data = torch.tensor(encode(text), dtype=torch.long)
print(tokenizer.encode('You are brilliant').ids)
print(tokenizer.encode('You are brilliant').tokens)
print(tokenizer.decode(tokenizer.encode('You are brilliant').ids))

print(data.shape, data.dtype)
#exit()

# separate data into train and testing
# e.g., 9-1 split, 90% training
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel(vocab_size, n_embed, block_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # learning rate 10**-3

for steps in range(iterations):
    # sample a batch of data to train
    xb, yb = get_batch(train_data, 'train', batch_size, block_size)
    
    # print intermediate output
    if steps % eval_interval == 0:
        losses = estimate_loss(model, train_data, val_data, batch_size,
                block_size, eval_iters)
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save model to local
outpath = os.path.join(indir, 'train.model')
if os.path.exists(outpath):
    idx = 1
    while os.path.exists(outpath + f".{idx}"):
        idx += 1
    torch.save(model.state_dict(), outpath + f".{idx}")
else:
    torch.save(model.state_dict(), outpath)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(model.generate(context, max_tokens=max_tokens)[0].tolist()))
for nxt_token in model.generate(context, max_tokens=max_tokens):
    print(tokenizer.decode(nxt_token[0].tolist(), skip_special_tokens=True), end='')
print('')
