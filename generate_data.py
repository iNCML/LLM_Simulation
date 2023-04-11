import numpy as np

nIntention = 1 #6
nLetters = 3

epsilon = 0.0 #1.0e-10

chars = '\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPGRSTUVWXYZ'

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


def message2intention(message):
  index = encode(message)
  probs = np.ones(nIntention)
  for c in range(nIntention):
    i = index[0]
    for j in index[1:]:
      probs[c] += np.log(TextTranMatrix[c,i,j])
      i = j

#      print(f'{c} {i} {j} {TextTranMatrix[c,i,j]}')
      
  return np.argmax(probs)


np.random.seed(1713)

IntentionMatrix = np.zeros((nIntention,nIntention))

for i in range(nIntention):
#  prob = np.random.rand(1)
  prob = 0.5
  IntentionMatrix[i,i] = prob 
  IntentionMatrix[i,(i+1)%nIntention] = 1.0-prob 


position = range(nLetters*nIntention)
intent = 0

TextTranMatrix = np.zeros((nIntention,nLetters*nIntention,nLetters*nIntention))+epsilon

for i in position[:nIntention]:
  tt = np.random.rand(nLetters,nLetters)
  tt = tt / np.expand_dims(np.sum(tt,-1),-1)  
  TextTranMatrix[i,i*nLetters:(i+1)*nLetters,i*nLetters:(i+1)*nLetters] = tt

#TextTranMatrix = TextTranMatrix/np.expand_dims(np.sum(TextTranMatrix,-1),-1)

correct = 0
total = 0

for i in range(1):
  intent= position[:nIntention] @ np.random.multinomial(1,IntentionMatrix[intent])
  ch = intent*nLetters
  output = ''
  for j in range(2000000):
    ch = position @ np.random.multinomial(1, TextTranMatrix[intent,ch])
    output= output + chars[ch+1]
  #print(f'intention={intent}: message={output}')
  print(output)
#  intent2 = message2intention(output) 
#  if intent == intent2:
#      correct += 1 
#  total = total + 1
#
#print(f'correct rate = {correct/total*100:.5f}%')

np.save('/tmp/Transition',TextTranMatrix)

#for c in range(nIntention):
#    for i in range(nLetters*nIntention):
#        for j in range(nLetters*nIntention):
#            print(f'{c} {i} {j} {TextTranMatrix[c,i,j]}')  
