import pandas as pd
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
import matplotlib.pyplot as plt
from fairclip_dataloader import build_bimodal_dataloaders
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """Text encoder for clinical notes using BERT"""
    def __init__(self, bert_model='bert-base-uncased', dropout=0.1):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_data):
        bert_outputs = self.bert(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask'])
        text_features = bert_outputs.last_hidden_state[:, 0]  
        text_features = self.dropout(text_features)
        
        return text_features
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class GlaucomaBiModel(nn.Module):
    def __init__(self, output_dim=2, proj_dim=256, fusion_heads=4, fusion_layers=4,
                dropout=0.1, vit_variant='google/vit-base-patch16-224-in21k', pretrained=True):
        super(GlaucomaBiModel, self).__init__()
        self.proj_dim = proj_dim
        if pretrained:
            self.fundus_encoder = ViTModel.from_pretrained(vit_variant)
        else:
            config = ViTConfig.from_pretrained(vit_variant)
            self.fundus_encoder = ViTModel(config)
        
        vit_feature_dim = self.fundus_encoder.config.hidden_size
        
        self.fundus_proj = nn.Sequential(
           nn.Linear(vit_feature_dim, proj_dim),
           nn.LayerNorm(proj_dim),
           nn.Dropout(dropout)
       )
        self.text_encoder = TextEncoder(dropout=dropout)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.bert_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout)
        )
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=fusion_heads,
            dim_feedforward=proj_dim*4,
            dropout=dropout,
            activation='gelu'
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_encoder_layer, 
            num_layers=fusion_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, output_dim)
        )
    
    def forward(self, fundus, text_data):
        fundus_outputs = self.fundus_encoder(fundus)
        fundus_features = fundus_outputs.pooler_output
        fundus_features = self.fundus_proj(fundus_features)
        text_features = self.text_encoder(text_data)  
        text_features = self.text_proj(text_features)
        combined_features = torch.stack([fundus_features, text_features], dim=0)
        fused_features = self.fusion_transformer(combined_features) 
        final_features = fused_features[0] 
        output = self.classifier(final_features)
        modality_features = [fundus_features, text_features]
        
        return output, modality_features
class Classifier(nn.Module):
   def __init__(self, in_dim, out_dim, num_heads=5, layers=2,
                relu_dropout=0.1, embed_dropout=0.3,
                attn_dropout=0.25, res_dropout=0.1):
       super(Classifier, self).__init__()
       self.classifier = nn.Sequential(
           nn.Linear(in_dim, in_dim),
           nn.ReLU(),
           nn.Dropout(relu_dropout),
           nn.Linear(in_dim, out_dim)
       )
   
   def forward(self, x):
       return self.classifier(x)

class ClassifierGuided(nn.Module):
    
    def __init__(self, output_dim, num_mod=2, proj_dim=128, num_heads=5, layers=2,  
                relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, attn_dropout=0.25):
       super(ClassifierGuided, self).__init__()
       self.num_mod = num_mod 
       self.prev_auc = [0.5] * self.num_mod
       self.classifiers = nn.ModuleList([
           Classifier(in_dim=proj_dim, out_dim=output_dim, layers=layers,
                      num_heads=num_heads, attn_dropout=attn_dropout, res_dropout=res_dropout,
                      relu_dropout=relu_dropout, embed_dropout=embed_dropout)
           for _ in range(self.num_mod)
       ])
     
      
    def cal_coeff(self,dataset, y, cls_res):
        """Calculate AUC for each modality classifier, preserving previous values on failure"""
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
                    print(f"Only one class present for modality {i}, using previous AUC: {self.prev_auc[i]:.4f}")
                    auc_list.append(self.prev_auc[i])
                    
            except Exception as e:
                print(f"AUC calculation failed for modality {i}, using previous AUC: {self.prev_auc[i]:.4f}. Error: {e}")
                auc_list.append(self.prev_auc[i])
                
        return auc_list
    
    def pairwise_logistic_surrogate_auc(self, probs, labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_scores = probs[pos_mask]
        neg_scores = probs[neg_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return torch.tensor(0.5, device=probs.device, dtype=probs.dtype)  # Neutral AUC
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
                group_mask = (demographics == group_val)
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