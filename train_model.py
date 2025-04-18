import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(522170)

# check cuda availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        eval_iters=200):
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
Bigram model
"""
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()

        self.block_size = block_size
        self.n_embed = n_embed

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        
        # positional embedding table (for each position in the context)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        
        # use a linear layer to obtain the final logits from the embedding
        self.linear = nn.Linear(n_embed, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape   # (B, T)

        # x and y are both of size (B, T) --> B the batch_size, T the number of positions
        tok_emb = self.token_embedding(x) # (B, T, n_embed)
        pos_emb = self.positional_embedding(torch.arange(T, device=device)) # (B, T, n_embed)
        new_x = tok_emb + pos_emb   # (B, T, n_embed)
        logits = self.linear(new_x)   # (B, T, vocab_size)

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
            x_cond = x[:, -self.block_size:]
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
