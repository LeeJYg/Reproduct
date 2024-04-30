import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from torch import nn, einsum

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device, offset = 0):
        seq = torch.arange(max_seq_len, device = device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        if values.dim() == 5:
            value_len, key_len = values.shape[1]*values.shape[2]*values.shape[3], keys.shape[1]*keys.shape[2]*keys.shape[3]
        else:
            value_len, key_len = values.shape[1]*values.shape[2], keys.shape[1]*keys.shape[2]

        if query.dim() == 5:
            query_len = query.shape[1]*query.shape[2]*query.shape[3]
        else:
            query_len = query.shape[1]*query.shape[2]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Query dot Key
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
    
class ChunkedCrossAttention(nn.Module):
    def __init__(self, embed_size, heads, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.cross_attention = SelfAttention(embed_size, heads)
    
    def forward(self, query, key, value, chunk_masks=None):
        # Assuming key and value are from the encoder and have the same shape
        batch_size, num_chunks, _, _ = key.shape
        chunked_key = key.view(batch_size, num_chunks, self.chunk_size, -1)
        chunked_value = value.view(batch_size, num_chunks, self.chunk_size, -1)
        
        chunk_key = torch.split(chunked_key, 1, dim=1)
        chunk_value = torch.split(chunked_value, 1, dim=1)
        chunk_query = torch.split(query, 1, dim=1)

        # Perform cross-attention on each chunk
        output = []
        for i in range(num_chunks):
            if chunk_masks is not None:
                chunk_mask = chunk_masks[:, i]
            else:
                chunk_mask = None
            attn_output = self.cross_attention(chunk_value[i], chunk_key[i], chunk_query[i], chunk_mask)
            output.append(attn_output)

        # Concatenate the output from all chunks
        output = torch.cat(output, dim=1)

        return output

class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout, is_cross=False):
        super(EncoderLayer, self).__init__()
        self.is_cross = is_cross
        self.self_attention = SelfAttention(embed_size, heads)
        if is_cross:
            self.cross_attention = SelfAttention(embed_size, heads) # 예시로 SelfAttention 사용, 실제로는 CrossAttention 필요
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = PositionwiseFeedforward(embed_size, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, retrieved_passages):
        chunk_num = value.shape[1]
        chunk_len = value.shape[3]
        
        if value.dim() == 5:
            value1 = value[:, :, 0, :, :].reshape(query.shape[0], chunk_num, chunk_len, -1)
            value2 = value[:, :, 1, :, :].reshape(query.shape[0], chunk_num, chunk_len, -1)
            
            self_att1 = self.self_attention(value1, value1, value1, mask)
            self_att2 = self.self_attention(value2, value2, value2, mask)
            
            self_att1 = self_att1.reshape(query.shape[0], query.shape[1], query.shape[2], -1)
            self_att2 = self_att2.reshape(query.shape[0], query.shape[1], query.shape[2], -1)
            
            self_att = self_att1 + self_att2
        else:
            self_att = self.self_attention(value, value, value, mask)
            self_att = self_att.reshape(query.shape[0], query.shape[1], query.shape[2], -1)

        x = self.dropout(self.norm1(self_att + query))

        if self.is_cross:
            cross_att = self.cross_attention(x, x, query, mask)
            cross_att = cross_att.reshape(query.shape[0], query.shape[1], query.shape[2], -1)
            x = self.dropout(self.norm2(cross_att + x))

        # Feedforward
        forward = self.feed_forward(x)
        out = self.dropout(self.norm3(forward + x))
        return out

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout, cross_attention=False, chunk_size=16):
        super().__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.chunked_cross_attention = ChunkedCrossAttention(embed_size, heads, chunk_size) if cross_attention else None
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size) if cross_attention else None
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = PositionwiseFeedforward(embed_size, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, src_mask, trg_mask, chunk_masks=None, encoder=None, retrieved_passages=None):
        attention = self.self_attention(query, query, query, trg_mask)
        attention = attention.reshape(query.shape[0], query.shape[1], query.shape[2], -1)
        
        query = self.dropout(self.norm1(attention + query))

        if retrieved_passages is not None and self.chunked_cross_attention is not None:
            enc_out = encoder(attention, src_mask, retrieved_passages)
            attention = self.chunked_cross_attention(query, enc_out, enc_out, chunk_masks)
            attention = attention.reshape(query.shape[0], query.shape[1], query.shape[2], -1)
            query = self.dropout(self.norm2(attention + query))
            
        out = self.feed_forward(query)
        out = self.dropout(self.norm3(out + query))
        return out
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, ff_dim, dropout, max_length, enc_cross_layer):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = RotaryEmbedding(embed_size)

        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, ff_dim, dropout, True if i in enc_cross_layer else False) for i in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, retrieved_passages):
        chunk_size = x.shape[-2]
        retrieved_len = retrieved_passages.shape[-1]
        
        q_positions = self.position_embedding(chunk_size, device=self.device)
        k_positions = self.position_embedding(retrieved_len, device=self.device)
        
        out = self.dropout(x + q_positions)
        ret_out = self.dropout(self.word_embedding(retrieved_passages) + k_positions)

        for layer in self.layers:
            out = layer(ret_out, ret_out, out, mask, retrieved_passages)

        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, cross_attention_layers, heads, device, ff_dim, dropout, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size, device=self.device)
        self.position_embedding = RotaryEmbedding(embed_size)
        self.layers = nn.ModuleList()

        for layer_num in range(1, num_layers + 1):
            cross_attention = (layer_num in cross_attention_layers)
            self.layers.append(DecoderLayer(embed_size, heads, ff_dim, dropout, cross_attention=cross_attention))

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, trg_mask, encoder=None, retrieved_passages=None):
        chunk_num, chunk_len = x.shape[-2], x.shape[-1]
        seq_len = chunk_num * chunk_len
        
        positions = self.position_embedding(seq_len, device=self.device)
        positions = positions.to(self.device)
        positions = positions.view(1, chunk_num, chunk_len, -1)
        
        x = self.dropout(self.word_embedding(x) + positions)
        #x는 1, 32, 512, 256 

        for layer in self.layers:
            x = layer(x, src_mask, trg_mask, encoder=encoder, retrieved_passages=retrieved_passages)

        out = self.fc_out(x)

        return out
    
class RETRO(nn.Module):
    def __init__(self, pad_id=0, num_token=97, embed_size=256, enc_num_layers=6, dec_num_layers=12, decoder_cross_attention_layer=(3, 6, 9), enc_cross_layer = (0,), forward_expansion=4, heads=8, dropout=0.1, device="cuda", max_length=512):
        super(RETRO, self).__init__()
        self.encoder = Encoder(num_token, embed_size, enc_num_layers, heads, device, forward_expansion * embed_size, dropout, max_length, enc_cross_layer=enc_cross_layer)
        self.decoder = Decoder(num_token, embed_size, dec_num_layers, decoder_cross_attention_layer, heads, device, forward_expansion * embed_size, dropout, max_length)
        self.src_pad_idx = pad_id
        self.trg_pad_idx = pad_id
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_num, trg_len = trg.shape
        seq_len = trg_num * trg_len
        trg_mask = torch.tril(torch.ones((seq_len, seq_len))).expand(N, 1, seq_len, seq_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg, retrieved_passages):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        out = self.decoder(src, src_mask, trg_mask, self.encoder, retrieved_passages)
        return out