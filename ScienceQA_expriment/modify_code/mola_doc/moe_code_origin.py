
from collections.abc import Mapping
from distutils.util import strtobool
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence
import torch
from huggingface_hub import Repository, create_repo
from packaging import version
from torch import nn
from transformers import __version__
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    nested_detach,
)
from tqdm import tqdm

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
#     model.train()
#     inputs = self._prepare_inputs(inputs)
#     labels = inputs["labels"]
#     print("labels min/max:", labels.min().item(), labels.max().item())
#     print("valid:", (labels!=-100).sum().item(), "/", labels.numel())
#     print("row0:", labels[0, :40].tolist())
#     print("unique row0:", sorted(set(labels[0].tolist()))[:20])
#     if is_sagemaker_mp_enabled():
#         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
#         return loss_mb.reduce_mean().detach().to(self.args.device)

#     with self.compute_loss_context_manager():
#         loss = self.compute_loss(model, inputs)

#     print("LOSS type:", type(loss),
#         "reqgrad:", loss.requires_grad,
#         "grad_fn:", loss.grad_fn)
#     print("is_grad_enabled:", torch.is_grad_enabled())

#     if self.args.n_gpu > 1:
#         loss = loss.mean()  # mean() to average on multi-gpu parallel training
#     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
#         # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
#         loss = loss / self.args.gradient_accumulation_steps

#     if self.do_grad_scaling:
#         self.scaler.scale(loss).backward()
#     elif self.use_apex:
#         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
#             scaled_loss.backward()
#     elif self.deepspeed:
#         # loss gets scaled under gradient_accumulation_steps in deepspeed
#         loss = self.deepspeed.backward(loss)
#     else:
#         loss.backward()
#     return loss.detach()

# def compute_loss(self, model, inputs, return_outputs=False):
#     if self.label_smoother is not None and "labels" in inputs:
#         labels = inputs.pop("labels")
#     else:
#         labels = None
#     outputs = model(**inputs)
#     print("grad_enabled (inside compute_loss):", torch.is_grad_enabled())
#     print("is_inference_mode:", torch.is_inference_mode_enabled())
#     if isinstance(outputs, dict) and "logits" in outputs:
#         print("LOGITS reqgrad:", outputs["logits"].requires_grad, "grad_fn:", outputs["logits"].grad_fn)

#     print(inputs.keys())
#     print("labels exists:", "labels" in inputs)
#     if "labels" in inputs:
#         print("labels dtype/shape:", inputs["labels"].dtype, inputs["labels"].shape)
#         print("labels unique sample:", torch.unique(inputs["labels"][:1]).tolist()[:20])

#     print("MODEL loss reqgrad:", (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).requires_grad,
#         "grad_fn:", (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).grad_fn)
#     # print(type(outputs))
#     # print(outputs.loss, outputs.loss.requires_grad if outputs.loss is not None else None)
#     # Save past state if it exists
#     # TODO: this needs to be fixed and made cleaner later.
#     if self.args.past_index >= 0:
#         self._past = outputs[self.args.past_index]

#     if labels is not None:
#         if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#             loss = self.label_smoother(outputs, labels, shift_labels=True)
#         else:
#             loss = self.label_smoother(outputs, labels)
#     else:
#         if isinstance(outputs, dict) and "loss" not in outputs:
#             raise ValueError(
#                 "The model did not return a loss from the inputs, only the following keys: "
#                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#             )
#         # We don't use .loss here since the model may return tuples instead of ModelOutput.
#         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#     return (loss, outputs) if return_outputs else loss

# def prediction_step(
#         self,
#         model: nn.Module,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#     has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
#     return_loss = inputs.get("return_loss", None)
#     if return_loss is None:
#         return_loss = self.can_return_loss
#     loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

#     inputs = self._prepare_inputs(inputs)
#     if ignore_keys is None:
#         if hasattr(self.model, "config"):
#             ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
#         else:
#             ignore_keys = []

#     # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
#     if has_labels or loss_without_labels:
#         labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
#         if len(labels) == 1:
#             labels = labels[0]
#     else:
#         labels = None

#     with torch.no_grad():
#         print("prediction called")
#         if is_sagemaker_mp_enabled():
#             raw_outputs = smp_forward_only(model, inputs)
#             if has_labels or loss_without_labels:
#                 if isinstance(raw_outputs, dict):
#                     loss_mb = raw_outputs["loss"]
#                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
#                 else:
#                     loss_mb = raw_outputs[0]
#                     logits_mb = raw_outputs[1:]

#                 loss = loss_mb.reduce_mean().detach().cpu()
#                 logits = smp_nested_concat(logits_mb)
#             else:
#                 loss = None
#                 if isinstance(raw_outputs, dict):
#                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
#                 else:
#                     logits_mb = raw_outputs
#                 logits = smp_nested_concat(logits_mb)
#         else:
#             if has_labels or loss_without_labels:
#                 with self.compute_loss_context_manager():
#                     loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#                 loss = loss.mean().detach()

#                 if isinstance(outputs, dict):
#                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
#                 else:
#                     logits = outputs[1:]
#             else:
#                 loss = None
#                 with self.compute_loss_context_manager():
#                     outputs = model(**inputs)
#                 if isinstance(outputs, dict):
#                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
#                 else:
#                     logits = outputs
#                 # TODO: this needs to be fixed and made cleaner later.
#                 if self.args.past_index >= 0:
#                     self._past = outputs[self.args.past_index - 1]

#     if prediction_loss_only:
#         return (loss, None, None)
#     logits = nested_detach(logits)
#     if len(logits) == 1:
#         logits = logits[0]
#     return (loss, logits, labels)

def load_balancing_loss_func(
        gate_logits: torch.Tensor, 
        num_experts: torch.Tensor = None, 
        top_k=2
    ) -> float:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device

    overall_loss = torch.tensor(0, dtype=torch.float32).to(compute_device)
    for layer_i in range(len(gate_logits)):
        layer_gate = gate_logits[layer_i]
        concatenated_gate_logits = layer_gate.to(compute_device)

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k[layer_i], dim=-1)

        # treat `top_k` as tokens (shape is `top_k X [batch_size X sequence_length]`)
        selected_experts = selected_experts.reshape(-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts[layer_i])
        expert_mask_1 = torch.sum(expert_mask, dim=-2)
        tokens_per_expert = expert_mask_1 / torch.sum(expert_mask_1)
        # expert_mask = torch.max(expert_mask, dim=-2).values
        #
        # # Compute the percentage of tokens routed to each experts
        # tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
        # overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
        layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts[layer_i]
        overall_loss += layer_loss
    return overall_loss

def train_moe_multiple_layers(
    model_list: List[torch.nn.Module],
    train_loader,
    optimizer,
    scheduler,
    device,
    num_epochs: int = 100,
    aux_weight: float = 0.01,            # auxiliary loss 가중치 (필요시 튜닝)
    num_experts: Optional[Sequence[int]] = None,  # layer별 expert 수, 예: [8]*32
    top_k: Optional[Sequence[int]] = None,        # layer별 topk, 예: [2]*32
    debug_every: int = 0,                 # 0이면 off, 예: 200이면 200 step마다 디버그
):
    for model in model_list:
        model.train()

    criterion = torch.nn.MSELoss()
    print(f"Running optimizer for all models together......")

    use_aux = (aux_weight is not None and aux_weight > 0.0)
    if use_aux:
        if num_experts is None or top_k is None:
            raise ValueError("aux_weight>0이면 num_experts와 top_k를 layer별 리스트로 넘겨야 합니다.")
        if len(num_experts) < len(model_list) or len(top_k) < len(model_list):
            raise ValueError("num_experts/top_k 길이는 model_list 길이 이상이어야 합니다.")

    global_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_main = 0.0
        running_aux = 0.0

        with tqdm(
            total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ) as pbar:
            for inputs_list, targets_list in train_loader:
                inputs_list = [x.to(device) for x in inputs_list]
                targets_list = [y.to(device) for y in targets_list]

                optimizer.zero_grad()
                # Forward pass
                outputs_list = []
                router_logits_tuple = []  # layer별 router logits 모아 tuple로 aux 계산

                ################################################
                for layer_i, (model, x) in enumerate(zip(model_list, inputs_list)):
                    # 가능한 다양한 forward 시그니처를 흡수
                    out = None
                    try:
                        out = model(x, return_router_logits=True)
                    except TypeError:
                        out = model(x)

                    if isinstance(out, tuple) and len(out) == 2:
                        y_pred, router_logits = out
                        outputs_list.append(y_pred)
                        router_logits_tuple.append(router_logits)
                    else:
                        # router_logits를 못 받으면 aux는 사실상 학습 불가
                        outputs_list.append(out)
                        # aux를 쓴다고 했는데 logits를 못 받는 모델이 있으면 즉시 에러로 잡는 게 안전
                        if use_aux:
                            raise RuntimeError(
                                f"Layer {layer_i} model forward가 (y, router_logits)를 반환하지 않습니다. "
                                "aux loss를 쓰려면 router_logits가 필요합니다."
                            )
                ################################################

                # 1) main loss (기존 방식 그대로: layer별 MSE 합)
                main_loss = sum(
                    criterion(y_hat, y)
                    for y_hat, y in zip(outputs_list, targets_list)
                )

                # 2) aux loss (Switch-style load balancing)
                if use_aux:
                    # load_balancing_loss_func는 gate_logits가 tuple일 때만 계산하도록 되어있음
                    gate_logits = tuple(router_logits_tuple)
                    aux_loss = load_balancing_loss_func(
                        gate_logits=gate_logits,
                        num_experts=list(num_experts[:len(model_list)]),
                        top_k=list(top_k[:len(model_list)]),
                    )
                else:
                    aux_loss = torch.tensor(0.0, device=device)

                loss = main_loss + aux_weight * aux_loss

                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach().item())
                running_main += float(main_loss.detach().item())
                running_aux += float(aux_loss.detach().item()) if use_aux else 0.0

                global_step += 1
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    main=f"{main_loss.item():.4f}",
                    aux=f"{aux_loss.item():.4f}" if use_aux else "0.0000",
                )

            if scheduler is not None:
                scheduler.step()

        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_main = running_main / max(len(train_loader), 1)
        epoch_aux = running_aux / max(len(train_loader), 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Main(MSE): {epoch_main:.4f} | Aux: {epoch_aux:.4f} | aux_w={aux_weight}")
    
    print("MoE Training completed!")
    return model_list
