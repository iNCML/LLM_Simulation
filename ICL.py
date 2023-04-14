# evaluate in-context learning under various number of prompts 

import math
import torch
import numpy as np
from torch.nn import functional as F
import argparse

from myModel import *
from preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model_file", type=str, help="the path for the model file")
parser.add_argument("-T", "--trans_file", type=str, help="the path for the transition file")
parser.add_argument("-n", "--eval_num", type=int, help="number of evaluated prompts")
parser.add_argument("-m", "--max_prompt", type=int, help="maximum number of evaluated prompts at each case")

args = parser.parse_args()

if args.model_file != None:
    model_file = args.model_file
else:
    model_file = '/tmp/my_model'

if args.trans_file != None:
    tran_file = args.trans_file
else:
    tran_file = '/tmp/Transition.npy'

if args.eval_num != None:
    eval_samples = args.eval_num
else:
    eval_samples = 200

if args.max_prompt != None:
    max_prompt = args.max_prompt
else:
    max_prompt = 4
    
# load model
print(f'loading model from {model_file} ...')
model = torch.load(model_file)
model.eval()
model.to('cpu')
block_size = model.config.block_size

print(f'block_size = {block_size}')

# load true transition distributions
print(f'loading transition distributions {tran_file} ...')
Transition = np.load(tran_file)
Transition = torch.from_numpy(Transition)

IntentionMatrix = np.load('/tmp/IntentTransition.npy')


np.set_printoptions(suppress = True)

def eval_transformer(prompt, model):
    X = torch.tensor(encode(prompt))
    X = X.view(1,-1)
    logits, _ = model(X)
    probs = F.softmax(logits.view(-1), dim=-1)
    return probs.detach().numpy()

intents = np.random.randint(len(Transition), size=eval_samples)
results = np.zeros((eval_samples, max_prompt))

for k in range(eval_samples):
    it = intents[k]

    # true prob dist
    nLetters = Transition.size(1)//Transition.size(0)
    arraysize = Transition.size(1)
    trueprob = np.zeros(arraysize+1)
    for i in range(2*nLetters):
        trueprob[1+(i+it*nLetters)%arraysize] = 1.0/(2*nLetters)
                                   
    prompt = ""
    for j in range(max_prompt):
        mes, ch = gen_message(Transition, it, 20)
        prompt = mes + "\n" + prompt
        probs = eval_transformer(prompt, model)
        results[k,j] = KL_Divergence(trueprob,probs)
    
    #print(results[k])

print(np.average(results,0))
