from config import *
from data import vocab_size
import torch
import math
import torch.nn as nn
import numpy as np

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()
        self.tok_embed = nn.Embedding(vocab_size,d_model)
        self.seg_embed = nn.Embedding(n_segments,d_model)
        self.pos_embed = nn.Embedding(maxlen,d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,input_ids,seg_ids):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len,dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(input_ids)
        embedding = self.tok_embed(input_ids)+self.pos_embed(pos)+self.seg_embed(seg_ids)
        return embedding


class ScaledDotProductionAttention(nn.Module): # 感觉这个注意力函数可以用函数实现，不用单独创建一个类
    def __init__(self):
        super(ScaledDotProductionAttention,self).__init__()

    def forward(self,Q,K,V,attn_mask):
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask,1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k*n_heads)
        self.W_K = nn.Linear(d_model,d_k*n_heads)
        self.W_V = nn.Linear(d_model,d_k*n_heads)

    def forward(self,Q,K,V,attn_mask):
        residual,batch_size = Q,Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)

        context = ScaledDotProductionAttention()(q_s,k_s,v_s,attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size,-1,n_heads*d_v)
        output = nn.Linear(n_heads*d_v,d_model)(context)
        return nn.LayerNorm(d_model)(output+residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)  # [bach_size, seq_len, d_model]  [batch_size, maxlen,d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, dim = 1, index=masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf

