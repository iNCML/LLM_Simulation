# Define all parts in a GPT model, such as attention, MLP, LN modules
#  (adapted from nanoGPT, https://github.com/karpathy/nanoGPT)

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    #Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class mySelfAttentionFunc(torch.autograd.Function):
    # X:[B, d, T];  M:[3*nh*hs, d]; W:[d, d]
    # return: Z: [B, ns*nh, T]
    def forward(ctx, X, M, W, conf_paras):
        mask = conf_paras['mask']
        n_embd = conf_paras['n_embd']
        n_head = conf_paras['n_head']
        B, d, T = X.size() # batch size,  embedding dimensionality (n_embd), block size
        assert d == n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        Q, K, V  = (M @ X).split(n_embd, dim=1)
        K = K.view(B, n_head, d // n_head, T)   # (B, nh, hs, T)
        Q = Q.view(B, n_head, d // n_head, T)   # (B, nh, hs, T)
        V = V.view(B, n_head, d // n_head, T)   # (B, nh, hs, T)

        att = (Q.transpose(-2, -1) @ K) * (1.0 / math.sqrt(Q.size(-2)))
        if conf_paras['causal']:
          att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))          
        att = F.softmax(att, dim=-2)   # column-wise softmax
        Z = V @ att   # (B, nh, hs, T) x (B, nh, T, T) -> (B, nh, hs, T)

        Z = Z.contiguous().view(B, d, T)  # re-assemble all head outputs side by side
        ctx.save_for_backward(X, M, Q, K, V, att, W, Z)      

        return W @ Z     # output projection

    # grad_y:[B, ns*nh, T]
    # return: grad_x:[B, d, T] , grad_M:[3*ns*nh,d]
    def backward(ctx, grad_y):
      X, M, Q, K, V, att, W, Z = ctx.saved_tensors
      B, n, h, T = V.size()      

      # error signal E: [B, nh, hs, T]
      E = W.transpose(-2,-1) @ grad_y
      E = E.view(B, n, h, T)  

      W_grad = torch.sum(grad_y @ Z.transpose(-2,-1), dim=0)
              
      VE = torch.transpose(V, -2,-1) @ E    # V':[B, ns, T, hs] x E:[B, nh, hs, T] ==> [B, nh, T, T]
       # S:[B, nh, T, T] = att \otimes VE
      S = (att*VE-torch.einsum('bnit,bnt->bnit', att,torch.einsum('bnit,bnit->bnt', att, VE))) * (1.0 / math.sqrt(h))

      A_grad = K @ S.transpose(-2,-1)    # K:[B, nh, hs, T] x S:[B, nh, T, T ] ==> [B, nh, hs, T]
      B_grad = Q @ S                     # Q:[B, nh, hs, T] x S:[B, nh, T, T ] ==> [B, nh, hs, T]
      C_grad = E @ att.transpose(-2,-1)  # E:[B, nh, hs, T] x att:[B, nh, T, T] ==> [B, nh, hs, T]
      # P: [B, 3*nh*hs, T]      
      P = torch.cat((A_grad,B_grad,C_grad),dim=1).view(B, 3*n*h, T)   
                  
       # [B, 3*nh*hs, T] x [B, d, T] ==> [3*nh*hs,d]  
      M_grad = torch.einsum('bit,bjt->ij', P, X)
      
       # M': [d, 3*nh*hs] x P:[B, 3*nh*hs, T] ==> [B, d, T]
      grad_x = M.transpose(0,1) @ P 

      X.grad = grad_x if X.requires_grad else None
      M.grad = M_grad if M.requires_grad else None
      W.grad = W_grad if W.requires_grad else None
        
      return grad_x, M_grad, W_grad, None

class myCausalSelfAttention(nn.Module):

    def __init__(self, config):
      super().__init__()
      assert config.n_embd % config.n_head == 0

      # create my attension function 
      self.myatt = mySelfAttentionFunc()

      # key, query, value projections for all heads, but in a batch
      self.M = (torch.rand((3*config.n_embd, config.n_embd),requires_grad=True) - 0.5) * (1.0/math.sqrt(config.n_embd))
      # output projection
      self.W = (torch.rand((config.n_embd, config.n_embd),requires_grad=True) - 0.5) * (1.0/math.sqrt(config.n_embd)) 
      self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).transpose(0,1).\
                           view(1, 1, config.block_size, config.block_size))
      self.mask = self.mask.to(config.device) 
      self.M = self.M.to(config.device)
      self.W = self.W.to(config.device)

      self.n_head = config.n_head
      self.n_embd = config.n_embd

    def forward(self, x):
        conf_paras = {}
        conf_paras['mask'] = self.mask
        conf_paras['n_embd'] = self.n_embd
        conf_paras['n_head'] = self.n_head
        conf_paras['causal'] = True

        x = x.transpose(-2,-1)
        z = self.myatt.apply(x, self.M, self.W, conf_paras)

        return z.transpose(-2,-1)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).transpose(0,1)
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, C // self.n_head, self.n_head).transpose(1, 3) # (B, nh, T, hs)
        q = q.view(B, T, C // self.n_head, self.n_head).transpose(1, 3) # (B, nh, T, hs)
        v = v.view(B, T, C // self.n_head, self.n_head).transpose(1, 3) # (B, nh, T, hs)

        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = (q.transpose(-2, -1) @ k) * (1.0 / math.sqrt(q.size(-2)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-2)
        att = self.attn_dropout(att)
        y = v @ att #@ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 3).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        #self.attn = myCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str = 'cpu'

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        # Return the number of parameters in the model.
 
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
  
        extra_args = dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
