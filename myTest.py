# Test distribution gap between true transitional distributions and the pre-trained transformers

import math 
import torch
import numpy as np
from torch.nn import functional as F
import argparse

from myModel import *
from preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--input_file", type=str, help="the path for the input file")
parser.add_argument("-M", "--model_file", type=str, help="the path for the model file")
parser.add_argument("-T", "--trans_file", type=str, help="the path for the transition distribution file")
parser.add_argument("-n", "--eval_samples", type=int, help="number of samples used in evaluation")
parser.add_argument("-t", "--seen", type=bool,  help="use training samples for evaluation")

args = parser.parse_args()

if args.input_file != None:
    input_file = args.output_file
else:
    input_file = '/tmp/input.txt'

if args.model_file != None:
    model_file = args.model_file
else:
    model_file = '/tmp/my_model'

if args.trans_file != None:
    tran_file = args.trans_file
else:
    tran_file = '/tmp/Transition.npy'

if args.eval_samples != None:
    eval_samples = args.eval_samples
else:
    eval_samples = 100

if args.seen != None:
    split = 'train'
else:
    split = 'val'
    
# load text file as a character string; build char-level vocabulary, encoder and decoder
print(f'loading data from {input_file} ...')
with open(input_file, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

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

SEP = encode('\n') 

# load model
print(f'loading model from {model_file} ...')
model = torch.load(model_file)
model.eval()
model.to('cpu')
block_size = model.config.block_size

# load true transition distributions
print(f'loading transition distributions {tran_file} ...')
Transition = np.load(tran_file)
Transition = torch.from_numpy(Transition)
nIntention = Transition.size(0)
nLetters = Transition.size(1)//Transition.size(0)

# poor man's data loader
def get_batch(split):
    data = data_tr if split == 'train' else data_val 
    i = torch.randint(len(data) - block_size,(1,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64))])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))])
    return x, y

ave_diff = 0.0

for k in range(eval_samples):
    X, Y = get_batch(split)
    logits, loss = model(X, Y)

    X = X.view(-1)
    Y = Y.view(-1)
    probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)

    score1 = 0.0
    for j in range(block_size):
        score1 += torch.log(probs[j,Y[j]])
        #print(f'transformer {X[j]-1} {Y[j]-1} {torch.log(probs[j,Y[j]])}')

    score2 = 0.0
    allscores = torch.zeros(nIntention)
    for j in range(block_size):
        if X[j] in SEP:
            allscores += math.log(1.0/nLetters)
            #print(f'true2:  {X[j]-1} {Y[j]-1} {math.log(1.0/nLetters)}')
        elif Y[j] in SEP:
            score2 += torch.logsumexp(allscores,0)
            allscores = torch.zeros(nIntention)
            #print(f'true2:  {X[j]-1} {Y[j]-1} 0.0')
        else:
            for k in range(nIntention):
                allscores[k] += torch.log(Transition[k,X[j]-1,Y[j]-1])
                #if (Transition[k,X[j]-1,Y[j]-1]>0.0):
                    #print(f'true:  {X[j]-1} {Y[j]-1} {torch.log(Transition[k,X[j]-1,Y[j]-1])}')
          
    score2 += torch.logsumexp(allscores,0)

    #print(f'transformer={score1/block_size:.5f}  true={score2/block_size:.5f}  diff={abs(score1-score2)/block_size:.5f}')
    ave_diff += abs(score1-score2)/block_size

print(f'average distribution difference = {ave_diff/eval_samples:.5f} (from {split})...')
                                          
                                          
