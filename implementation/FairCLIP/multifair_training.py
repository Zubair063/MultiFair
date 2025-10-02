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
from models_module import GlaucomaBiModel, ClassifierGuided
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import gc
import random
    


def eval_glaucoma(results, truths):
   test_preds = results.contiguous().view(-1, 2).cpu().detach().numpy()
   test_truth = truths.contiguous().view(-1).cpu().detach().numpy()
   test_preds_i = np.argmax(test_preds, axis=1)
   test_truth_i = test_truth
 
   probs = test_preds[:, 1] 
   auc = roc_auc_score(test_truth_i, probs)
  
   metrics = {'auc': auc}
  
   return metrics


def multifair(train_loader, valid_loader, 
                        num_epochs=30, lr=1e-4, cls_lr=5e-5,
                        proj_dim=128, num_heads=4, layers=4, cls_layers=2,
                        rou=1.3, lambda_gm=0.1, fairness_threshold=0.08, 
                        fairness_coeff=0.8, fairness_modulation=0.6, clip=0.8,
                        relu_dropout=0.1, embed_dropout=0.2, res_dropout=0.1, 
                        out_dropout=0.1, attn_dropout=0.2, pretrained=True, 
                        save_path="demo.pth"):
    """
    
    Args:
        train_loader: DataLoader for training data (fundus, text, demographics, labels)
        valid_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate for main model
        cls_lr: Learning rate for classifiers
        proj_dim: Projection dimension
        num_heads: Number of attention heads
        layers: Number of transformer layers
        cls_layers: Number of classifier layers
        rou: Scaling hyperparameter for modality modulation
        lambda_gm: Weight for gradient modulation loss
        fairness_threshold: Threshold for fairness gap to trigger fairness-aware gradient modulation
        fairness_coeff: Weight for fairness term in loss
        fairness_modulation: Factor to scale the gradient modulation based on fairness
        clip: Gradient clipping threshold
        *_dropout: Various dropout probabilities
        pretrained: Whether to use pretrained weights
        save_path: Path to save the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GlaucomaBiModel( 
        output_dim=2,
        proj_dim=proj_dim,
        fusion_heads=num_heads,
        fusion_layers=layers,
        dropout=0.1,
        pretrained=pretrained 
    ).to(device)
    
    classifier = ClassifierGuided(
        output_dim=2, 
        num_mod=2,
        proj_dim=proj_dim, 
        num_heads=num_heads, 
        layers=cls_layers, 
        relu_dropout=relu_dropout, 
        embed_dropout=embed_dropout,
        res_dropout=res_dropout, 
        attn_dropout=attn_dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=6e-3)
    cls_optimizer = optim.AdamW(classifier.parameters(), lr=cls_lr, weight_decay=3e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    auc_prev = [0.5, 0.5] 
    l_gm = 0 
    best_valid_auc = 0
    

    ema_decay = 0.9
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
       
        'val_auc': [],
        'fairness_gap': [],
        'fairness_priority': []  
    }
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        model.train()
        classifier.train()
        epoch_loss = 0
        epoch_cggm_loss = 0
        batch_count = 0
        fairness_priority_count = 0  

        train_preds = []
        train_truths = []
        epoch_fairness_gaps = []
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs} [Train]")
            
        for batch_data in train_loader:
            try:
                if len(batch_data) == 4:
                    fundus_data, text_data, demographics, batch_Y = batch_data
                    batch_demo = demographics.squeeze(1).to(device)
                else:
                    fundus_data, text_data, batch_Y = batch_data
                    batch_demo = None
                
                fundus_data = fundus_data.to(device)
                text_data = {k: v.to(device) for k, v in text_data.items()}
                batch_Y = batch_Y.to(device)
                
                batch_size = fundus_data.size(0)
                batch_count += 1
                cls_optimizer.zero_grad()
                
                with torch.no_grad():
                    _, hs = model(fundus_data, text_data)
                
                cls_res = classifier(hs)
                
                cls_grad = []
                for i in range(2):
                    cls_loss = criterion(cls_res[i], batch_Y)
                    scaler.scale(cls_loss).backward(retain_graph=(i < 1))

                    for name, para in classifier.named_parameters():
                        if f'classifiers.{i}.classifier.3.weight' in name:
                            if para.grad is not None:
                                cls_grad.append(para.grad.clone())
          
                scaler.step(cls_optimizer)
                scaler.update()

                auc_current = classifier.cal_coeff('glaucoma', batch_Y, cls_res)
                diff = [auc_current[i] - auc_prev[i] for i in range(2)] 
                diff_sum = sum(abs(d) for d in diff) + 1e-8
                coeff = [(diff_sum - abs(d)) / diff_sum for d in diff]
                auc_prev = auc_current.copy() 
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    preds, hs = model(fundus_data, text_data)
                    train_preds.append(preds.detach().cpu())
                    train_truths.append(batch_Y.detach().cpu())
                    task_loss = criterion(preds, batch_Y)
                
                    if len(cls_grad) >= 2: 
                        fusion_grad = None
                        for name, para in model.named_parameters():
                            if 'classifier.weight' in name:
                                fusion_grad = para.grad
                                break
                        if fusion_grad is None:
                            dummy_loss = criterion(preds, batch_Y)
                            dummy_loss.backward(retain_graph=True)
                            for name, para in model.named_parameters():
                                if 'classifier.weight' in name:
                                    fusion_grad = para.grad
                                    break
                            optimizer.zero_grad()
                        
    
                        if fusion_grad is not None:
                            sim1 = torch.nn.functional.cosine_similarity(
                                fusion_grad.flatten().unsqueeze(0), 
                                cls_grad[0].flatten().unsqueeze(0)
                            )
                            sim2 = torch.nn.functional.cosine_similarity(
                                fusion_grad.flatten().unsqueeze(0), 
                                cls_grad[1].flatten().unsqueeze(0)
                            )
                    
                            l_gm = sum(abs(c) for c in coeff) - (coeff[0] * sim1 + coeff[1] * sim2)
                            l_gm = l_gm / 2.0
                        else:
                            l_gm = torch.tensor(0.0, device=device)
                    else:
                        l_gm = torch.tensor(0.0, device=device)
                    
                    if batch_demo is not None:
                        auc_dict = classifier.cal_coeff_de(batch_Y, cls_res, batch_demo)

                        if not hasattr(classifier, "ema_auc_de"):
                            classifier.ema_auc_de = {i: {'male': 0.5, 'female': 0.5} for i in auc_dict}

                        for i in auc_dict:
                            for group in ['male', 'female']:
                                prev_ema = classifier.ema_auc_de[i][group]
                                classifier.ema_auc_de[i][group] = (
                                    ema_decay * prev_ema + (1 - ema_decay) * auc_dict[i][group]
                                )

                        avg_auc = {i: (classifier.ema_auc_de[i]['male'] + classifier.ema_auc_de[i]['female']) / 2.0 for i in auc_dict}
                  
                        fairness_vals = torch.tensor([
                            torch.abs(classifier.ema_auc_de[i][group] - avg_auc[i])
                            for i in auc_dict
                            for group in ['male', 'female']
                        ], device='cpu') 

                        fairness_gap = fairness_vals.mean().item()
                        epoch_fairness_gaps.append(fairness_gap)
                        fairness_mod_factors = {
                            i: {
                                group: 1 + fairness_modulation * (avg_auc[i] - classifier.ema_auc_de[i][group]) / max(1e-4, fairness_threshold)
                                for group in ['male', 'female']
                            }
                            for i in auc_dict
                        }

                        group_counts = {
                            group: np.sum((batch_demo.squeeze().detach().cpu().numpy() == val))
                            for group, val in zip(['male', 'female'], [1.0, 0.0])
                        }
                        total_samples = sum(group_counts.values()) if sum(group_counts.values()) else 1
                        group_probs_fraction = {group: group_counts[group] / total_samples for group in group_counts}
                        fairness_batch_factor = {
                            i: sum(group_probs_fraction[group] * fairness_mod_factors[i][group] for group in group_counts)
                            for i in auc_dict
                        }

                        fusion_group_aucs = {}
                        for group_val, group_name in zip([1.0, 0.0], ['male', 'female']):
                            group_indices = (batch_demo == group_val).nonzero(as_tuple=True)[0]
                            if len(group_indices) > 1:
                                probs = torch.softmax(preds[group_indices], dim=1)[:, 1].detach().cpu().numpy()
                                targets_np = batch_Y[group_indices].detach().cpu().numpy()
                                if len(set(targets_np)) > 1:
                                    auc = roc_auc_score(targets_np, probs)
                                else:
                                    auc = 0.5
                                fusion_group_aucs[group_name] = auc

                        if len(fusion_group_aucs) == 2:
                            fusion_fairness_gap = abs(max(fusion_group_aucs.values()) - min(fusion_group_aucs.values()))
                            fairness_priority = fusion_fairness_gap > fairness_threshold
                            if fairness_priority:
                                fairness_priority_count += 1
                        else:
                            fairness_priority = False
                    else:
                        fairness_gap = 0
                        fairness_priority = False
                        fairness_batch_factor = {0: 1.0, 1: 1.0} 
                        
                    if fairness_priority:
                        total_loss = task_loss + lambda_gm * l_gm + fairness_coeff * fairness_gap
                    else:
                        total_loss = task_loss + lambda_gm * l_gm
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
        
                if fairness_priority:
                    for name, params in model.named_parameters():
                        if params.grad is not None:
                            if 'fundus_encoder' in name:
                                params.grad *= coeff[0] * rou * fairness_batch_factor[0]
                            elif 'text_encoder' in name:
                                params.grad *= coeff[1] * rou * fairness_batch_factor[1]
                else:
                    for name, params in model.named_parameters():
                        if params.grad is not None:
                            if 'fundus_encoder' in name:
                                params.grad *= coeff[0] * rou
                            elif 'text_encoder' in name:
                                params.grad *= coeff[1] * rou
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                
                del fundus_data, text_data, batch_Y, preds, hs, cls_res
                if batch_demo is not None:
                    del batch_demo
                torch.cuda.empty_cache()
                
                pbar.set_postfix(
                    loss=f"{task_loss.item():.4f}"
                )
                pbar.update(1)
                
            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                import traceback
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
                continue
        
        pbar.close()
        
        avg_epoch_loss = epoch_loss / (batch_count * batch_size) if batch_count > 0 else 0
        
        history['train_loss'].append(avg_epoch_loss)
        
        train_preds.clear()
        train_truths.clear()
        gc.collect()
        torch.cuda.empty_cache()
        
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        results = []
        truths = []
        val_pbar = tqdm(total=len(valid_loader), desc=f"Epoch {epoch}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for val_data in valid_loader:
                if len(val_data) == 4:
                    fundus_data, text_data, _, batch_Y = val_data
                else:
                    fundus_data, text_data, batch_Y = val_data
                    
                fundus_data = fundus_data.to(device)
                text_data = {k: v.to(device) for k, v in text_data.items()}
                batch_Y = batch_Y.to(device)
                
                try:
                    preds, _ = model(fundus_data, text_data)
                    loss = criterion(preds, batch_Y)
                    val_loss += loss.item() * fundus_data.size(0)
                    val_batch_count += 1
                    results.append(preds.cpu())
                    truths.append(batch_Y.cpu())
                    del fundus_data, text_data, batch_Y, preds
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
                val_pbar.update(1)
        val_pbar.close()
        
        if results:
            results_cat = torch.cat(results)
            truths_cat = torch.cat(truths)
            
            metrics = eval_glaucoma(results_cat, truths_cat)
            val_auc = metrics['auc']
            
            avg_val_loss = val_loss / (val_batch_count * valid_loader.batch_size) if val_batch_count > 0 else 0
            history['val_loss'].append(avg_val_loss)
            scheduler.step(avg_val_loss)
            end = time.time()
            duration = end - start
            
            print(f"Epoch {epoch}/{num_epochs} | Time: {duration:.2f}s | "
                  f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_valid_auc:
                best_valid_auc = val_auc
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with validation AUC: {val_auc:.4f}")
                
            del results_cat, truths_cat
            results.clear()
            truths.clear()
            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            print(f"Epoch {epoch}: No valid results for validation")
            
    return model, history





def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
   
    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
   
    num_epochs = 10
    model_save_path = "Multifair_fairvision.pth"
   
    try:
        DATA_ROOT = "/FairCLIP"
        CSV_PATH  = "/FairCLIP/data_summary.csv"

        train_loader, val_loader, test_loader = build_bimodal_dataloaders(
            data_root=DATA_ROOT,
            csv_path=CSV_PATH,
            batch_size=32,
            num_workers=4,
            demographic_key="gender" )
    
        try:
            model, metrics = multifair(
                train_loader=train_loader,
                valid_loader=val_loader,
                num_epochs=num_epochs,
                lr=2e-5,  
                cls_lr=1e-5, 
                proj_dim=256, 
                num_heads=8,   
                layers=3,      
                cls_layers=2,  
                rou=1.2,       
                lambda_gm=0.1, 
                fairness_threshold=0.05,
                fairness_coeff=0.5, 
                fairness_modulation=0.5,
                clip=0.8,      
                relu_dropout=0.1,
                embed_dropout=0.2,
                res_dropout=0.1,
                out_dropout=0.1,
                attn_dropout=0.2, 
                pretrained=True,
                save_path = model_save_path
                )
            print("\nTraining completed successfully!")


                
        except Exception as e:
            print(f"Error during bimodal training: {str(e)}")
            import traceback
            traceback.print_exc()
           
    except Exception as e:
        print(f"Error setting up bimodal data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()