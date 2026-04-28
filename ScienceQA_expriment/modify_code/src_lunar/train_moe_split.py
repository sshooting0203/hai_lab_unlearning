# train_moe_split.py
# 둘이 분리해서 수렴하는 속도를 다르게 해야함 
# learning rate를 다르게 설정하기(이걸로)
import torch
from tqdm import tqdm
from typing import List, Optional, Sequence
import torch.nn.functional as F

def set_moe_trainable(model_list, mode='both'):
    """
    mode: 'expert' (LoRA만), 'router' (Gate만), 'both' (둘 다)
    """
    for model in model_list:
        # 1. Router Gate
        if hasattr(model, "gate"):
            # MLP(Sequential)인 경우 처리
            if isinstance(model.gate, torch.nn.Sequential):
                # 앞단 Hidden Layer들은 항상 Freeze (사용자 요청 사항)
                for i in range(len(model.gate) - 1):
                    for param in model.gate[i].parameters():
                        param.requires_grad = False
                
                # 마지막 Output Layer만 학습 여부 결정
                for param in model.gate[-1].parameters():
                    param.requires_grad = (mode in ['router', 'both'])
            else:
                # Linear Gate인 경우 기존 방식 유지
                for param in model.gate.parameters():
                    param.requires_grad = (mode in ['router', 'both'])
        
        # 2. LoRA Experts
        if hasattr(model, "experts"):
            for param in model.experts.parameters():
                param.requires_grad = (mode in ['expert', 'both'])
        
        # 3. Base (주로 고정되어 있음)
        if hasattr(model, "linear"):
            for param in model.linear.parameters():
                param.requires_grad = False

def zero_expert_param_grads(model, freeze_indices):
    if not freeze_indices:
        return
    with torch.no_grad():
        if hasattr(model, "lora_A") and model.lora_A.grad is not None:
            for idx in freeze_indices:
                model.lora_A.grad[idx].zero_()
        if hasattr(model, "lora_B") and model.lora_B.grad is not None:
            for idx in freeze_indices:
                model.lora_B.grad[idx].zero_()

def zero_gate_row_grads(model, freeze_indices):
    if not freeze_indices:
        return
    with torch.no_grad():
        gate = model.gate
        # MLP(Sequential)인 경우 마지막 층을 타겟팅
        target_layer = gate[-1] if isinstance(gate, torch.nn.Sequential) else gate
        
        if hasattr(target_layer, "weight") and target_layer.weight.grad is not None:
            for idx in freeze_indices:
                target_layer.weight.grad[idx].zero_()
        if hasattr(target_layer, "bias") and target_layer.bias is not None and target_layer.bias.grad is not None:
            for idx in freeze_indices:
                target_layer.bias.grad[idx].zero_()

def train_moe_split_phases(
    model_list: List[torch.nn.Module],
    train_loader,
    device,
    current_forget_idx: int,
    expert_epochs: int = 10,
    router_epochs: int = 10,
    expert_lr: float = 1e-4,
    router_lr: float = 1e-3,
    max_grad_norm: Optional[float] = None,
    debug_every: int = 10,
):
    mse_loss_fn = torch.nn.MSELoss()
    ce_loss_fn  = torch.nn.CrossEntropyLoss()
    
    frozen_expert_indices = [0] + list(range(1, current_forget_idx))
    frozen_gate_indices   = list(range(1, current_forget_idx))

    # ---------------------------------------------------------
    # Phase 1: Expert Training
    # ---------------------------------------------------------
    if expert_epochs > 0:
        print(f"\n[Split Train] Starting Phase 1: Expert Training ({expert_epochs} epochs, lr={expert_lr})")
        set_moe_trainable(model_list, mode='expert')
        
        expert_params = []
        for model in model_list:
            if hasattr(model, "experts"):
                expert_params.extend([p for p in model.experts.parameters() if p.requires_grad])
        
        if expert_params:
            optimizer_e = torch.optim.AdamW(expert_params, lr=expert_lr)
            
            for epoch in range(expert_epochs):
                running_mse = 0.0
                desc = f"Expert Phase Epoch [{epoch+1}/{expert_epochs}]"
                with tqdm(total=len(train_loader), desc=desc) as pbar:
                    for batch in train_loader:
                        inputs_list, targets_list, domain_labels = batch
                        domain_labels = domain_labels.to(device)
                        inputs_list = [x.to(device) for x in inputs_list]
                        targets_list = [y.to(device) for y in targets_list]

                        forget_mask = (domain_labels == current_forget_idx)
                        if not forget_mask.any():
                            pbar.update(1)
                            continue
                        
                        optimizer_e.zero_grad()
                        total_mse = torch.tensor(0.0, device=device)
                        
                        for model, x, y_tgt in zip(model_list, inputs_list, targets_list):
                            y_mix, _, BAx, base = model(x, return_all_experts=True)
                            y_forget_expert = base[forget_mask] + model.scaling * BAx[forget_mask, current_forget_idx]
                            total_mse = total_mse + mse_loss_fn(y_forget_expert, y_tgt[forget_mask])
                        
                        total_mse.backward()
                        
                        for model in model_list:
                            zero_expert_param_grads(model, frozen_expert_indices)
                        
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(expert_params, max_grad_norm)
                        
                        optimizer_e.step()
                        running_mse += total_mse.item()
                        pbar.update(1)
                
                print(f"Phase 1 Epoch {epoch+1} | Avg MSE Loss: {running_mse/len(train_loader):.6f}")

    # ---------------------------------------------------------
    # Phase 2: Router Training
    # ---------------------------------------------------------
    if router_epochs > 0:
        print(f"\n[Split Train] Starting Phase 2: Router Training ({router_epochs} epochs, lr={router_lr})")
        set_moe_trainable(model_list, mode='router')
        
        router_params = []
        for model in model_list:
            if hasattr(model, "gate"):
                router_params.extend([p for p in model.gate.parameters() if p.requires_grad])
        
        if router_params:
            optimizer_r = torch.optim.AdamW(router_params, lr=router_lr)
            
            for epoch in range(router_epochs):
                running_ce = 0.0
                desc = f"Router Phase Epoch [{epoch+1}/{router_epochs}]"
                with tqdm(total=len(train_loader), desc=desc) as pbar:
                    for batch in train_loader:
                        inputs_list, _, domain_labels = batch
                        domain_labels = domain_labels.to(device)
                        inputs_list = [x.to(device) for x in inputs_list]

                        optimizer_r.zero_grad()
                        total_ce = torch.tensor(0.0, device=device)
                        
                        for model, x in zip(model_list, inputs_list):
                            _, logits, _, _ = model(x, return_router_logits=True)
                            E = logits.size(-1)
                            
                            targets_router = domain_labels.clone()
                            valid_mask = (targets_router == 0) | (targets_router == current_forget_idx)
                            
                            if valid_mask.any():
                                logits_valid = logits[valid_mask].float()
                                targets_valid = targets_router[valid_mask]
                                
                                keep_mask = torch.zeros(E, dtype=torch.bool, device=logits.device)
                                keep_mask[0] = True
                                keep_mask[current_forget_idx] = True
                                masked_logits = logits_valid.masked_fill(~keep_mask.unsqueeze(0), -1e9)
                                total_ce = total_ce + ce_loss_fn(masked_logits, targets_valid)
                        
                        if total_ce.requires_grad and total_ce.item() > 0:
                            total_ce.backward()
                            
                            for model in model_list:
                                zero_gate_row_grads(model, frozen_gate_indices)
                            
                            if max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(router_params, max_grad_norm)
                            
                            optimizer_r.step()
                            running_ce += total_ce.item()
                        pbar.update(1)
                
                print(f"Phase 2 Epoch {epoch+1} | Avg CE Loss: {running_ce/len(train_loader):.6f}")

    print("\n[Split Train] All Phases Completed!")
    return model_list
