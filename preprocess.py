import numpy as np
import torch
from torch.nn import functional as F

### about alphabet ###
chars = '\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPGRSTUVWXYZ'

# create a mapping from characters to integers                                                                                                  
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers                                                             
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


def gen_message(Transition, intention, length):
    nIntention = Transition.size(0)
    nLetters = Transition.size(1)//Transition.size(0)
    position = range(nLetters*nIntention)
    
    ch = intention*nLetters + position[:nLetters]  @ np.random.multinomial(1, np.ones(nLetters)/nLetters)
    message = chars[ch+1]
    for j in range(length):
        ch = position @ np.random.multinomial(1, Transition[intention,ch])
        message = message + chars[ch+1]
    
    return message, ch

def message2intention(Transition, message):
    nIntention = len(Transition)
    probs = np.ones(nIntention)

    index = encode(message)
    for c in range(nIntention):
        i = index[0]
        for j in index[1:]:
            probs[c] *= Transition[c,i-1,j-1]
            i = j
    return np.argmax(probs)

def KL_Divergence(p1,p2):
    return np.inner(p1, np.log(p1+1.0e-10)-np.log(p2+1.0e-10))


def eval_transformer(prompt, model):
    X = torch.tensor(encode(prompt))
    X = X.view(1,-1)
    
    logits, _ = model(X)
    probs = F.softmax(logits.view(-1), dim=-1)
    return probs.detach().numpy()
                    
