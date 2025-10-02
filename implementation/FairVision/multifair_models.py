__all__ = ['Linear', 'LayerNorm', 'SinusoidalPositionalEmbedding', 'MultiheadAttention', 'TransformerEncoderLayer', 'TransformerEncoder', 'GlaucomaViTModel', 'Classifier', 'ClassifierGuided']

import os
import gc
import random
import torch
import math
import time
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from libauc.metrics import auc_roc_score
from transformers import ViTModel, ViTConfig
from oct_fundus_dataloader import OCTTransform, GlaucomaOCTFundusDataset, build_oct_fundus_dataloaders
import math

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.register_buffer('weights', None)
        self.register_buffer('positions_buffer', None)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.device
        if self.weights is None or self.weights.size(0) < max_pos or self.weights.device != device:
            weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx).to(device)
            self.register_buffer('weights', weights, persistent=False)
        if self.positions_buffer is None or self.positions_buffer.size(1) < seq_len or self.positions_buffer.device != device:
            positions = torch.arange(0, max_pos, device=device)[None, :].expand(bsz, -1)
            if positions.size(1) > seq_len:
                positions = positions[:, :seq_len]
            self.register_buffer('positions_buffer', positions, persistent=False)
        else:
            positions = self.positions_buffer[:, :seq_len]
            if positions.size(0) != bsz:
                positions = positions[0:1, :].expand(bsz, -1)
                self.register_buffer('positions_buffer', positions, persistent=False)
        positions = positions.long()
        return self.weights[positions].reshape(bsz, seq_len, -1)

    def max_positions(self):
        return int(100000.0)

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** (-0.5)
        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        q = q.contiguous().reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().reshape(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return (attn, attn_weights)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout, attn_mask=attn_mask)
            self.layers.append(new_layer)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        intermediates = [x]
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)
        if self.normalize:
            x = self.layer_norm(x)
        return x

class GlaucomaViTModel(nn.Module):

    def __init__(self, output_dim=2, proj_dim=256, fusion_heads=4, fusion_layers=4, dropout=0.1, vit_variant='google/vit-base-patch16-224-in21k', pretrained=True):
        super(GlaucomaViTModel, self).__init__()
        self.proj_dim = proj_dim
        if pretrained:
            self.fundus_encoder = ViTModel.from_pretrained(vit_variant)
        else:
            config = ViTConfig.from_pretrained(vit_variant)
            self.fundus_encoder = ViTModel(config)
        vit_feature_dim = self.fundus_encoder.config.hidden_size
        self.fundus_proj = nn.Sequential(nn.Linear(vit_feature_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout))
        if pretrained:
            self.oct_slice_encoder = ViTModel.from_pretrained(vit_variant)
        else:
            self.oct_slice_encoder = ViTModel(config)
        self.oct_aggregator = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=vit_feature_dim, nhead=4, dim_feedforward=vit_feature_dim * 4, dropout=dropout, activation='gelu'), num_layers=2)
        self.oct_proj = nn.Sequential(nn.Linear(vit_feature_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout))
        self.oct_pos_encoding = nn.Parameter(torch.zeros(1, 10, vit_feature_dim))
        nn.init.normal_(self.oct_pos_encoding, std=0.02)
        fusion_encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=fusion_heads, dim_feedforward=proj_dim * 4, dropout=dropout, activation='gelu')
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=fusion_layers)
        self.classifier = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj_dim, output_dim))

    def forward(self, fundus, oct):
        fundus_outputs = self.fundus_encoder(fundus)
        fundus_features = fundus_outputs.pooler_output
        fundus_features = self.fundus_proj(fundus_features)
        batch_size, num_slices, channel, height, width = oct.size()
        oct_reshaped = oct.view(batch_size * num_slices, channel, height, width)
        if channel == 1:
            oct_reshaped = oct_reshaped.repeat(1, 3, 1, 1)
        oct_slice_outputs = self.oct_slice_encoder(oct_reshaped)
        oct_slice_features = oct_slice_outputs.pooler_output
        oct_slice_features = oct_slice_features.view(batch_size, num_slices, -1)
        max_slices = min(num_slices, self.oct_pos_encoding.size(1))
        oct_slice_features = oct_slice_features[:, :max_slices, :] + self.oct_pos_encoding[:, :max_slices, :]
        oct_slice_features = oct_slice_features.permute(1, 0, 2)
        oct_aggregated = self.oct_aggregator(oct_slice_features)
        oct_features = oct_aggregated[0]
        oct_features = self.oct_proj(oct_features)
        combined_features = torch.stack([fundus_features, oct_features], dim=0)
        fused_features = self.fusion_transformer(combined_features)
        final_features = fused_features[0]
        output = self.classifier(final_features)
        modality_features = [fundus_features, oct_features]
        return (output, modality_features)

class Classifier(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=5, layers=2, relu_dropout=0.1, embed_dropout=0.3, attn_dropout=0.25, res_dropout=0.1):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Dropout(relu_dropout), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.classifier(x)

class ClassifierGuided(nn.Module):

    def __init__(self, output_dim, num_mod, proj_dim=128, num_heads=5, layers=2, relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, attn_dropout=0.25):
        super(ClassifierGuided, self).__init__()
        self.num_mod = num_mod
        self.classifiers = nn.ModuleList([Classifier(in_dim=proj_dim, out_dim=output_dim, layers=layers, num_heads=num_heads, attn_dropout=attn_dropout, res_dropout=res_dropout, relu_dropout=relu_dropout, embed_dropout=embed_dropout) for _ in range(self.num_mod)])
        self.prev_auc = [0.5] * self.num_mod

    def cal_coeff(self, dataset, y, cls_res):
        auc_list = []
        for i, output in enumerate(cls_res):
            try:
                probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                true_labels = y.detach().cpu().numpy()
                if len(set(true_labels)) > 1:
                    auc = roc_auc_score(true_labels, probs)
                    self.prev_auc[i] = auc
                    auc_list.append(auc)
                else:
                    print(f'Only one class present for modality {i}, using previous AUC: {self.prev_auc[i]:.4f}')
                    auc_list.append(self.prev_auc[i])
            except Exception as e:
                print(f'AUC calculation failed for modality {i}, using previous AUC: {self.prev_auc[i]:.4f}. Error: {e}')
                auc_list.append(self.prev_auc[i])
        return auc_list

    def pairwise_logistic_surrogate_auc(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_scores = probs[pos_mask]
        neg_scores = probs[neg_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return torch.tensor(0.5, device=probs.device, dtype=probs.dtype)
        diff = pos_scores[:, None] - neg_scores[None, :]
        surrogate_loss = torch.log1p(torch.exp(-diff))
        surrogate_auc = 1 - surrogate_loss.mean()
        return surrogate_auc

    def cal_coeff_de(self, y, cls_res, demographics):
        modalities = len(cls_res)
        group_values = [1.0, 0.0]
        group_names = ['male', 'female']
        auc_dict = {i: {} for i in range(modalities)}
        for i, output in enumerate(cls_res):
            probs = torch.softmax(output, dim=1)[:, 1]
            for group_val, group_name in zip(group_values, group_names):
                group_mask = demographics == group_val
                group_labels = y[group_mask]
                group_probs = probs[group_mask]
                if group_labels.numel() > 1 and torch.unique(group_labels).numel() > 1:
                    auc = self.pairwise_logistic_surrogate_auc(group_probs, group_labels.float())
                else:
                    auc = torch.tensor(0.5, device=probs.device, dtype=probs.dtype)
                auc_dict[i][group_name] = auc
        return auc_dict

    def forward(self, x):
        cls_outputs = []
        for i in range(len(x)):
            cls_outputs.append(self.classifiers[i](x[i]))
        return cls_outputs
