"""
TEST
"""

import torch
from myModel import *
import numpy as np
from torch.nn import functional as F

# load text file as a character string; build char-level vocabulary, encoder and decoder

with open('/tmp/input.txt', 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
#chars = sorted(list(set(data)))
#vocab_size = len(chars)
#print("all the unique characters:", ''.join(chars))
#print(f"vocab size: {vocab_size:,}")

chars = '\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPGRSTUVWXYZ'

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

data_tr  = np.array(train_ids, dtype=np.uint16)
data_val = np.array(val_ids, dtype=np.uint16)


# start to generate outputs

start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None #200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'

block_size = 64

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

model = torch.load('/tmp/my_model')

model.eval()
model.to(device)

Transition = np.load('/tmp/Transition.npy')
Transition = torch.from_numpy(Transition)
Transition.to(device)

nIntention = 1
nLetters = 3

for c in range(nIntention):
    for i in range(nLetters*nIntention):
        for j in range(nLetters*nIntention):
            print(f'{c} {i} {j} {torch.log(Transition[c,i,j])}')  



# poor man's data loader
def get_batch(split):
    data = data_tr if split == 'train' else data_val
 
    i = torch.randint(len(data) - block_size,(1,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64))])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))])
    #if device_type == 'cuda':
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

split = 'val'
eval_iters = 100

SEP = encode('\n') 

print(SEP) 


for k in range(eval_iters):
    X, Y = get_batch(split)

    logits, loss = model(X, Y)

    X = X.view(-1)
    Y = Y.view(-1)
    probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)

    score1 = 0.0
    for j in range(block_size):
        score1 += torch.log(probs[j,Y[j]])
 #       print(f'transformer {X[j]} {Y[j]} {torch.log(probs[j,Y[j]])}')

    score2 = 0.0
    allscores = torch.zeros(nIntention)
    for j in range(block_size):
        if X[j] == SEP:
            score2 += torch.logsumexp(allscores,0)
            allscores = torch.zeros(nIntention)
        elif Y[j] == SEP:
            None 
        else:
            for k in range(nIntention):
                allscores[k] += torch.log(Transition[k,X[j]-1,Y[j]-1])
#                print(f'true: {k} {X[j]-1} {Y[j]-1} {torch.log(Transition[k,X[j]-1,Y[j]-1])}')
          
    score2 += torch.logsumexp(allscores,0)

    print(f'transformer = {score1:.3f}  true = {score2:.3f}   diff={score1-score2:.3f}')
    
                                          
                                          
