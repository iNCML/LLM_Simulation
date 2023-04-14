# evaluate language understanding under one prompt 

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
parser.add_argument("-l", "--len_prompt", type=int, help="maximum number of evaluated prompts at each case")

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


if args.len_prompt != None:
    len_prompt = args.len_prompt
else:
    len_prompt = 5
    
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

results = np.zeros((eval_samples, len_prompt))

for ll in range(1, len_prompt+1):
    for k in range(eval_samples):
        
        it = np.random.randint(len(Transition), size=1).item()
        prompt, ch = gen_message(Transition, it, ll)        
        probs = eval_transformer(prompt, model)

        # true prob dist
        nLetters = Transition.size(1)//Transition.size(0)
        arraysize = Transition.size(1)
        trueprob = np.zeros(arraysize+1)
        trueprob[1:] = Transition[it,ch]  

#        print(f'prompt={prompt} KLD={KL_Divergence(trueprob,probs)}')
        results[k,ll-1] = KL_Divergence(trueprob,probs)
print(np.average(results,0))
