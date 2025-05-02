import os, sys
from tokenizers import Tokenizer, models, trainers
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(522170)

# check cuda availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# initialize a seed for reproducibility
batch_size = 64      # batch size
block_size = 256     # maximum context length for prediction
iterations = 5000    # number of iterations for training
eval_interval = 500  # print per X iterations
eval_iters = 100     # num iterations used for evaluation (to get an average)
max_tokens = 150     # maximum number of tokens to generate
n_embed    = 384     # embedding dimension
nhead      = 6       # number of heads for MHA
n_layers   = 5       # number of blocks/decoder layers to have
learning_rate = 3e-4 # learning rate
dropout    = 0.2     # dropout rate

"""
Function to split data into mini-batches
"""
def get_batch(data, split, batch_size, block_size):
    #data = train_data if split == "train" else val_data
    # random start point between [0, len(data), - block_size), do so for batch_size times
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # to(device) for both x and y
    x, y = x.to(device), y.to(device)
    return x, y

"""
Estimate loss on the current model. Do not run gradient
"""
@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size,
        eval_iters=500):
    out = {}
    model.eval()
    for split, data in {'train': train_data, 'val': val_data}.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # reset back to training mode
    model.train()
    return out

"""
Single head self-attention
"""
class Head(nn.Module):
    def __init__(self, head_size, context_size):
        super().__init__()
        # key, query, value linear projections
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # not a module and assigned as a "buffer" for masking
        self.register_buffer('tril', torch.tril(
            torch.ones(context_size, context_size)))
        
        # add dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None):
        B, T, C = x.shape   # (B, T, C)

        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # compute attention scores ("affinities" between tokens)
        weight = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        
        # apply dropout to the affinities
        weight = self.dropout(weight)
        
        # perform the weighted aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = weight @ v # (B, T, head_size)
        return out

"""
Multi-head attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, head_size, context_size):
        super().__init__()
        # reuse what we just implemented above
        self.heads = nn.ModuleList([Head(head_size, context_size) for _ in range(nhead)])
        # add a projection, linear transformation of the outcome of the MHA layer
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x, y=None):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

"""
A feedforward layer with linear connection
"""
class FeedForward(nn.Module):
    # with expansion for inner layer
    def __init__(self, n_embed, expansion_factor=4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, n_embed * expansion_factor),
            nn.ReLU(),
            nn.Linear(n_embed * expansion_factor, n_embed),
            nn.Dropout(dropout),     # dropout right before skip connect back to the input
        )
    def forward(self, x, y=None):
        return self.layer(x)

"""
A decoder block
"""
class DecoderBlock(nn.Module):
    def __init__(self, n_embed, nhead, context_size):
        super().__init__()
        # embedding dimension and number of heads
        
        # compute the head dimension to ensure we have the correct dimension
        head_size = n_embed // nhead
        
        # layernorm before attention 
        self.ln1 = nn.LayerNorm(n_embed)

        # MHA layer
        self.attention = MultiHeadAttention(nhead, head_size, context_size)
        
        # layernorm before feedforward
        self.ln2 = nn.LayerNorm(n_embed)

        # feed-forward linear layer, with expansion factor
        self.ffwd = FeedForward(n_embed)

    def forward(self, x, y=None):
        # add skip connect after attention
        # NOTE: deviate from the original Attention is All you need paper, apply layernorm BEFORE transformation with MHA
        x = x + self.attention(self.ln1(x))
        # also skip connect after feed-forward layer, and apply layernorm BEFORE transformation with ffwd
        x = x + self.ffwd(self.ln2(x))
        return x

"""
Bigram model
"""
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        
        # positional embedding table (for each position in the context)
        self.positional_embedding = nn.Embedding(context_size, n_embed)
        
        # have multiple decoder blocks and a layernorm after all decoder blocks
        self.decoders = nn.Sequential(*[DecoderBlock(n_embed, nhead, context_size) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)   # typically add a layernorm before the final output head
        
        # use a linear layer to obtain the final logits from the embedding
        self.output_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape   # (B, T)

        # x and y are both of size (B, T) --> B the batch_size, T the number of positions
        # convert input tokens to embedding
        tok_emb = self.token_embedding(x) # (B, T, n_embed)
        # add in positional information
        pos_emb = self.positional_embedding(torch.arange(T, device=device)) # (B, T, n_embed)
        # concatenate the two as input to following layers
        new_x = tok_emb + pos_emb   # (B, T, n_embed)

        # go over just decoder blocks
        new_x = self.decoders(new_x)

        # feed to final linear output head to get output vocab_size dimension, LayerNorm BEFORE getting to the head
        logits = self.output_head(self.ln(new_x))   # (B, T, vocab_size)

        # skip computing a loss
        if y is None:
            loss = None
        else:
            # add cross entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, x, max_tokens):
        # x.shape == (B, T), an array of tokens in the current context, for B batch items
        for _ in range(max_tokens):
            # crop the context to block_size to be able to feed to self
            x_cond = x[:, -block_size:]   # last block_size tokens
            # get the prediction based on x
            logits, loss = self(x_cond)
            # focus on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities for each vocab/token
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            nxt_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, nxt_token), dim=1)  # (B, T+1)
            yield nxt_token
        #return x

if __name__ == "__main__":
    # 1. model data dir
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
    #print(encode('You are brilliant'))
    #print(decode(encode('You are brilliant')))
    #data = torch.tensor(encode(text), dtype=torch.long)

    # use tokenizer bytepair encoding to train on the input corpus
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

    # encode text data with our newly trained BPE tokenizer
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    print(tokenizer.encode('You are brilliant').ids)
    print(tokenizer.decode(tokenizer.encode('You are brilliant').ids))

    # create a tensor object on data
    print(data.shape, data.dtype)

    # separate data into train and testing
    # e.g., 9-1 split, 90% training
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(vocab_size, block_size)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for steps in range(iterations):
        # sample a batch of data to train
        xb, yb = get_batch(data, 'train', batch_size, block_size)
        
        # evaluate current model training and validation loss
        if steps % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size,
                    block_size, eval_iters)
            print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # training, backprop
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save model to local, not overwriting existing ones
    outpath = os.path.join(indir, 'gpt2.model')
    if os.path.exists(outpath):
        idx = 1
        while os.path.exists(outpath + f".{idx}"):
            idx += 1
        torch.save(model.state_dict(), outpath + f".{idx}")
    else:
        torch.save(model.state_dict(), outpath)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for nxt_token in model.generate(context, max_tokens=max_tokens):
        print(tokenizer.decode(nxt_token[0].tolist(), skip_special_tokens=True), end='')
    print('')
