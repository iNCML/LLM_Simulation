### about alphabet ###
chars = '\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPGRSTUVWXYZ'

# create a mapping from characters to integers                                                                                                  
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers                                                             
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string                                                    
#def message2intention(message, TextTranMatrix):
#  index = encode(message)
#  probs = np.ones(nIntention)
#  for c in range(nIntention):
#    i = index[0]
#    for j in index[1:]:
#      probs[c] *= TextTranMatrix[c,i-1,j-1]
#      i = j
#  return np.argmax(probs)
