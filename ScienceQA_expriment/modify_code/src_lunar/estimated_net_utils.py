# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch, os 
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


class EstimatedNet(torch.nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias, 
        original_down_proj_weight, 
        if_mask=False
    ):
        super(EstimatedNet, self).__init__()

        # Define the layers with the same dimensions as the original MLP
        self.down_proj = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.original_down_proj_weight = original_down_proj_weight
        # self.rms_norm = RMSNorm(out_features)
        self.if_mask = if_mask
        # Initialize the weights randomly (default initialization in PyTorch)
        self._init_weights()

    def _init_weights(self):
        # nn.init.xavier_uniform_(self.down_proj.weight)
        with torch.no_grad():  # Disable gradient tracking while setting weights
            self.down_proj.weight.copy_(self.original_down_proj_weight)

    def forward(self, x, mask=None):
        # Forward pass based on the given architecture
        output = self.down_proj(x)
        if self.if_mask:
            output = output * mask
        return output


class LUNAR_LoRA_net(torch.nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        lora_rank=8, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        pretrained_weight=None
    ):
        super(LUNAR_LoRA_net, self).__init__()

        dtype = pretrained_weight.dtype if pretrained_weight is not None else torch.float32 # @ 수정 : fp16, bf16 conflict
        device = pretrained_weight.device if pretrained_weight is not None else 'cuda'
        # Define the original linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False, device=device, dtype=dtype)

        # Initialize the linear layer's weight with the pretrained weight if provided
        if pretrained_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_weight)

        # Freeze the original linear layer's weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Define LoRA parameters (low-rank adaptation matrices)
        self.lora_r = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.lora_A = torch.nn.Parameter(torch.randn(lora_rank, input_dim) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(output_dim, lora_rank) * 0.01)

    def forward(self, x):
        # print("x dtype:", x.dtype, "w dtype:", self.linear.weight.dtype) # x dtype: torch.float16, w dtype: torch.bfloat16
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        base_output = self.linear(x)
        x_d = self.lora_dropout(x)     
        lora_update = (x_d @ self.lora_A.t()) @ self.lora_B.t() 
        return base_output + self.scaling * lora_update
    
    # @ 수정 : gate 적용 가능하도록 
    # def forward(self, x, gate=None):
    #     if x.dtype != self.linear.weight.dtype:
    #         x = x.to(self.linear.weight.dtype)

    #     base_output = self.linear(x)
    #     x_d = self.lora_dropout(x)
    #     lora_update = (x_d @ self.lora_A.t()) @ self.lora_B.t()
    #     lora_update = self.scaling * lora_update

    #     gate = gate if gate is not None else getattr(self, "_gate_vec", None)

    #     if gate is None:
    #         return base_output + lora_update # gate가 설정되지 않으면 일반 forward
        
    #     B = base_output.size(0) # @ 수정 : Batch Size에 따라 gate 맞추기
    #     if gate.size(0) != B:
    #         if gate.size(0) > B:
    #             gate = gate[:B] # 마지막 배치가 더 작아지는 경우가 대부분 -> 앞에서 자르기
    #         else:
    #             print("gate.size(0) : ", gate.size(0), ", B : ",B)
    #             raise RuntimeError(f"Can't match batch size.")
                
    #     while gate.dim() < base_output.dim(): # gate가 설정되면 hook을 사용하여 batch 샘플에 lora update를 반영할지 말지 결정
    #         gate = gate.unsqueeze(-1)  # gate: [B] or [B,1] -> broadcast

    #     # print("base_output", base_output.shape, "lora_update", lora_update.shape, "gate", gate.shape, flush=True)
    #     return base_output + gate.to(base_output.dtype) * lora_update # base_output : torch.Size([16, 171, 4096]), lora_update : torch.Size([16, 171, 4096])

    def merge_weights(self):
        # Merge the LoRA weights into the linear layer's weight
        with torch.no_grad():
            merged_weight = self.linear.weight + (self.lora_B @ self.lora_A)#.T
            self.linear.weight.copy_(merged_weight)

        # After merging, LoRA parameters can optionally be deleted or frozen
        del self.lora_A
        del self.lora_B

def _get_down_proj(model, l):
    if hasattr(model, "model_block_modules"):
        return model.model_block_modules[l].mlp.down_proj
    raise TypeError(f"Can't find modules.")

def _set_down_proj(model, l: int, new_module):
    if hasattr(model, "model_block_modules"):
        model.model_block_modules[l].mlp.down_proj = new_module
        return
    raise TypeError(f"Can't set modules.")

# @ 수정 : LoRA_net 저장
def save_lora_only(
    model, 
    layer_idxs,
    save_path
):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    payload = {}
    for l in layer_idxs:
        m = _get_down_proj(model, l)
        if not isinstance(m, LUNAR_LoRA_net):
            raise TypeError(f"layer {l} down_proj is {type(m)}")
        payload[l] = {
            "lora_A": m.lora_A.detach().cpu(),
            "lora_B": m.lora_B.detach().cpu(),
            "lora_r": m.lora_r,
            "lora_alpha": m.lora_alpha,
            "lora_dropout": float(m.lora_dropout.p),
        }
    torch.save(payload, save_path)
    print(f"[SAVE] LUNAR LoRA-only -> {save_path}")


# updated_model = load_lora_only( # 이미 t이전에 저장된 lora model을 load 
#     updated_model,
#     load_path=cfg.load_unlearned_model_path, # load할 lora의 경로 
#     layer_idxs=layer_idx_list, # unlearn할 target layer
# )
# @ 수정 : LoRA_net 로드
def load_lora_only(
    model, 
    load_path, 
    layer_idxs=None, 
    device="cuda", 
    dtype=torch.bfloat16
):
    payload = torch.load(load_path, map_location="cpu")
    use_layers = layer_idxs if layer_idxs is not None else list(payload.keys())

    for l in use_layers:
        obj = payload[l]
        base = _get_down_proj(model, l)

        if isinstance(base, LUNAR_LoRA_net):
            base_w = base.linear.weight.detach().cpu()
        else: # nn.Linear
            base_w = base.weight.detach().cpu()

        lunar = LUNAR_LoRA_net(
            input_dim=base_w.shape[1],
            output_dim=base_w.shape[0],
            lora_rank=obj["lora_r"],
            lora_alpha=obj["lora_alpha"],
            lora_dropout=obj["lora_dropout"],
            pretrained_weight=base_w,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            lunar.lora_A.copy_(obj["lora_A"].to(device=device, dtype=dtype))
            lunar.lora_B.copy_(obj["lora_B"].to(device=device, dtype=dtype))
        _set_down_proj(model, l, lunar)
        
    print(f"[LOAD] LUNAR LoRA-only loaded from {load_path}")
    return model


class ActivationDataset_multiple_layers(Dataset):
    def __init__(self, inputs_list, targets_list, domain_label_list=None): # @@ : 수정 
        self.inputs_list = inputs_list
        self.targets_list = targets_list
        self.domain_label_list = domain_label_list

    def __len__(self):
        return self.inputs_list[0].size(0)

    def __getitem__(self, idx):
        # return self.inputs_list[idx], self.targets_list[idx]
        # return [inputs[idx] for inputs in self.inputs_list], [targets[idx] for targets in self.targets_list], self.domain_label_list[idx] # @@ : 수정
        return [inputs[idx] for inputs in self.inputs_list], [
            targets[idx] for targets in self.targets_list
        ]



class ActivationDataset_moe_multiple_layers(Dataset):
    def __init__(self, inputs_list, targets_list, domain_label_list=None): # @@ : 수정 
        self.inputs_list = inputs_list
        self.targets_list = targets_list
        self.domain_label_list = domain_label_list

    def __len__(self):
        return self.inputs_list[0].size(0)

    def __getitem__(self, idx):
        return [inputs[idx] for inputs in self.inputs_list], [targets[idx] for targets in self.targets_list], self.domain_label_list[idx] # @@ : 수정


# Train a single layer fully connected network(하나의 모델)
def train(
    model, train_loader, optimizer, scheduler, device, num_epochs=100
):
    model.train()  

    criterion = torch.nn.MSELoss()
    print(f"Running optimizer for one model......")
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(
            total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ) as pbar:
            # Loop through the batches of the training data
            for input, target in train_loader:
                input = input.to(device)
                target = target.to(device)

                optimizer.zero_grad()  # Zero the gradients
                # Forward pass
                outputs = model(input)
                loss = criterion(outputs, target)

                loss.backward()
                optimizer.step()

                # Keep track of the running loss for the current epoch
                running_loss += loss.item()

                # Update tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            if scheduler is not None:
                scheduler.step()

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training completed!")
    return model


# Train a single layer fully connected network(여러 개의 모델)
def train_multiple_layers(
    model_list, 
    train_loader, 
    optimizer, 
    scheduler, 
    device, 
    num_epochs=100
):
    for model in model_list:
        model.train()

    criterion = torch.nn.MSELoss()
    print(f"Running optimizer for all models together......")
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(
            total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ) as pbar:
            # Loop through the batches of the training data
            for inputs_list, targets_list in train_loader:
                inputs_list = [input.to(device) for input in inputs_list]
                targets_list = [target.to(device) for target in targets_list]

                optimizer.zero_grad()  # Zero the gradients
                # Forward pass
                outputs_list = [
                    model(inputs) for model, inputs in zip(model_list, inputs_list)
                ]
                # loss 계산 
                loss = sum(
                    [
                        criterion(outputs, target)
                        for outputs, target in zip(outputs_list, targets_list)
                    ]
                )

                loss.backward()
                optimizer.step()

                # Keep track of the running loss for the current epoch
                running_loss += loss.item()

                # Update tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            if scheduler is not None:
                scheduler.step()

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training completed!")
    return model_list
