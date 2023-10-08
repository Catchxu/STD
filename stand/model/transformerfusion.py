import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=False, attention_dropout=0.0):
        super(Attention, self).__init__()

        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)

        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        # self.w = nn.Linear(d_model, d_model, bias=False) 
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        from math import pi
        # queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        out = 4. / pi * self.dropout(torch.atan(1. / (torch.abs(queries - keys) + 1e-4))) * values
        #out = 4. / pi * self.dropout(torch.atan(self.w(torch.abs(queries - keys)))) * values
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=False, dropout=0.05, activation="gelu"):
        super(EncoderLayer, self).__init__()
        #d_ff = 4 * d_model
        self.attention = Attention(d_model=d_model, n_heads=n_heads, mask_flag=mask_flag)
        #self.fc1 = nn.Linear(d_model, d_ff)
        #self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, y):
        new_x = self.attention(
            x, y, y
        )#x y y 
        
        z = x + y + new_x
        
        #y = self.fc1(y)
        #y = self.fc2(x + y)
        return z#y


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model):
        super(Encoder, self).__init__()

        attn_layers = [encoder_layer for l in range(num_layers)]
        self.attn_layers = nn.ModuleList(attn_layers)
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, x, y):
        
        for attn_layer in self.attn_layers:
            z = attn_layer(x, y)

        #x = self.norm(x)
        return z