# script to generate training corpus 
#
import sys
import os
import argparse
import numpy as np
from preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--output_file", type=str, help="the path for the output file")
parser.add_argument("-m", "--nMessages", type=int, help="the number of generated messages")
parser.add_argument("-l", "--lenMessage", type=int, help="the length of each generated message")
parser.add_argument("-I", "--nIntention", type=int, help="the total number of distinct intentions")
parser.add_argument("-L", "--nLetters", type=int, help="the total number of distinct letters used for a message")
parser.add_argument("-e", "--epsilon", type=float, help="the added noise level")

args = parser.parse_args()

if args.output_file != None:
    output_file = args.output_file
else:
    output_file = '/tmp/input.txt'

if args.nMessages != None:
    nMessages = args.nMessages
else:
    nMessages = 100000

if args.lenMessage != None:
    lenMessage = args.lenMessage
else:
    lenMessage = 20

if args.nIntention != None:
    nIntention = args.nIntention
else:
    nIntention = 6

if args.nLetters != None:
    nLetters = args.nLetters
else:
    nLetters = 3

if args.epsilon != None:
    epsilon = args.epsilon
else:
    epsilon = 0.0

print(f'use {nIntention} distinct intentions {nLetters} distinct letters for each message eps={epsilon} ...')
    
### about alphabet ###
#chars = '\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPGRSTUVWXYZ'

# create a mapping from characters to integers
#stoi = { ch:i for i,ch in enumerate(chars) }
#itos = { i:ch for i,ch in enumerate(chars) }
#def encode(s):
#    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
#def decode(l):
#    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
#


def message2intention(message, TextTranMatrix):
  index = encode(message)
  probs = np.ones(nIntention)
  for c in range(nIntention):
    i = index[0]
    for j in index[1:]:
      probs[c] *= TextTranMatrix[c,i-1,j-1]
      i = j
  return np.argmax(probs)

### set up all transition matrix 

np.random.seed(1713)

IntentionMatrix = np.zeros((nIntention,nIntention))

for i in range(nIntention):
#  prob = np.random.rand(1)
  prob = 0.5
  IntentionMatrix[i,i] = prob 
  IntentionMatrix[i,(i+1)%nIntention] = 1.0-prob 

intent = 0  # initial intention 
position = range(nLetters*nIntention)

TextTranMatrix = np.zeros((nIntention,nLetters*nIntention,nLetters*nIntention))
for i in position[:nIntention]:
  tt = np.random.rand(nLetters,nLetters)
  tt = tt / np.expand_dims(np.sum(tt,-1),-1)  
  TextTranMatrix[i,i*nLetters:(i+1)*nLetters,i*nLetters:(i+1)*nLetters] = tt

# adding noises for ambiguous languages
if epsilon > 0.0:
    TextTranMatrix = TextTranMatrix + epsilon
    TextTranMatrix = TextTranMatrix/np.expand_dims(np.sum(TextTranMatrix,-1),-1)

#
# generate training corpus
#

correct = 0
total = 0

print(f'generate training corpus at {output_file} ({nMessages} messages and {lenMessage} letters each message) ...')
fout = open(output_file, 'w')

for i in range(nMessages):
  intent= position[:nIntention] @ np.random.multinomial(1,IntentionMatrix[intent])
  ch = intent*nLetters + position[:nLetters]  @ np.random.multinomial(1, np.ones(nLetters)/nLetters)
  output = chars[ch+1]
  for j in range(lenMessage):
    ch = position @ np.random.multinomial(1, TextTranMatrix[intent,ch])
    output= output + chars[ch+1]
  #print(f'intention={intent}: message={output}')
  #print(output)
  fout.write(output+'\n')
  intent2 = message2intention(output, TextTranMatrix) 
  correct = correct + 1 if intent == intent2 else correct
  total = total + 1

fout.close()

print(f'language ambiguity: {(1.0-correct/total):.5f}')

np.save('/tmp/Transition',TextTranMatrix)

#for c in range(nIntention):
#    for i in range(nLetters*nIntention):
#        for j in range(nLetters*nIntention):
#            print(f'{c} {i} {j} {TextTranMatrix[c,i,j]}')  
