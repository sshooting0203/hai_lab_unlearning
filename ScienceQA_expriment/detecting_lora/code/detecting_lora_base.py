import fire, torch, os, random, transformers, gc, torch, copy 
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig
from src.modeling_llama_hacked_o import LlamaForCausalLM_ood
from transformers import AutoConfig, Trainer, LlamaTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets, ClassLabel
from utils.prompter import Prompter
import numpy as np
from transformers import default_data_collator

os.environ["WANDB_DISABLED"] = "true"

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

################### pooling 관련 ################### 
def mean_pool(h, attention_mask):
    # h: [B,T,D], attention_mask: [B,T] (0/1)
    m = attention_mask.unsqueeze(-1).to(h.dtype)  # [B,T,1]
    denom = m.sum(dim=1).clamp(min=1.0)           # [B,1]
    return (h * m).sum(dim=1) / denom             # [B,D]

def last_k_mean_pool(h, attention_mask, k=8):
    # h: [B,T,D], attention_mask: [B,T]
    B, T, D = h.shape
    lengths = attention_mask.long().sum(dim=1)           # [B]
    idx = torch.arange(T, device=h.device).unsqueeze(0)  # [1,T]
    # 각 샘플마다 [len-k, len) 구간만 1
    start = (lengths - k).clamp(min=0).unsqueeze(1)      # [B,1]
    end = lengths.unsqueeze(1)                           # [B,1]
    m = ((idx >= start) & (idx < end) & (attention_mask.bool())).unsqueeze(-1)
    m = m.to(h.dtype)                                    # [B,T,1]
    denom = m.sum(dim=1).clamp(min=1.0)
    return (h * m).sum(dim=1) / denom

class AttnPool(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )
    def forward(self, h, attention_mask):
        # h: [B,T,D], attention_mask: [B,T]
        scores = self.scorer(h).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,T,1]
        return (h * w).sum(dim=1)  # [B,D]

################### class incremental loss ################### 
def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs, dim=1) # q_c
    tar = torch.nn.functional.softmax(targets, dim=1) # q_c'
    if exp != 1: # KD loss with temperature scaling
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)  # class에 대해 sum
    ce = -(tar * out.log()).sum(1) # −q_c'(x)log(q_c(x))
    if size_average:
        ce = ce.mean()
    return ce

def criterion(self, t, outputs, targets, targets_old):
    loss_dist = 0 # KD loss for all previous tasks 
    if t > 0: # 이전 클래스가 존재 
        loss_dist += cross_entropy(torch.cat(outputs[:t], dim=1),torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T)
    # trade-off - the lambda from the paper if lamb=-1
    if self.lamb == -1:
        lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device) 
        # self.task_cls = torch.tensor([head.out_features for head in self.heads])
        # C_old / C_t + C_old
        return (1.0 - lamb) * torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + lamb * loss_dist   
        # (1-lamb) x Lc + lamb x Ld
    else: 
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + self.lamb * loss_dist
        # self.lamb가 -1이면 설정한 self.lamb 사용

class EarlyLayerOODHead(nn.Module):
    def __init__(
        self, 
        llama_causal_ood, 
        use_layers=(1,2,3), 
        freeze_backbone=True,
        pool="last",
        eos_id=None,
        pad_id=None,
        debug=None,
    ):
        super().__init__()
        self.backbone = llama_causal_ood.model  # LlamaModel 사용한다고 보면 됨 
        self.use_layers = use_layers
        self.pool = pool
        D = llama_causal_ood.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(D * len(use_layers), D),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D, 2), # ID와 OOD 
        )
        if pool == "attn":
            self.attn_pool = AttnPool(D)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.debug = debug

    def forward(
            self, 
            input_ids, 
            attention_mask=None, 
            position_ids=None, 
            labels=None
        ):
        if attention_mask is None: # [B,T] [8,256]
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # last_idx = attention_mask.long().sum(dim=1) - 1  # [B]
        # last_idx = torch.clamp(last_idx, min=0) # 각 샘플마다 다른 “마지막 유효 토큰 위치”

        B, T = attention_mask.shape
        idx = torch.arange(T, device=attention_mask.device).unsqueeze(0).expand(B, -1)  # [B,T]
        # mask==0인 곳은 -1로 만들어서 max를 취하면 마지막 1의 인덱스가 됨
        last_idx = idx.masked_fill(attention_mask == 0, -1).max(dim=1).values  # [B]
        last_idx = last_idx.clamp(min=0)

        outputs, _o_loss = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            active_adapters_d = None, 
            ood_weight = None, 
            lora_start_layer = None, 
            olora = None, 
            use_cache=False,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        ) # hs = [3, 8, 256, 4096] 레이어 3개 x batch 8개 x token 256개 x hidden states 4096차원
        hs = outputs.hidden_states  # tuple: (emb, layer0, layer1, layer2, ...)
        ar = torch.arange(input_ids.size(0), device=input_ids.device)

        # picked_id = input_ids[ar, last_idx]  # [B]
        # eos_mask = (input_ids == self.eos_id) & (attention_mask.bool())  # [B,T]
        # has_eos = eos_mask.any(dim=1)  # [B]
        # rev_idx = torch.flip(eos_mask, dims=[1]).float().argmax(dim=1)  # [B]
        # last_eos_idx = (input_ids.size(1) - 1) - rev_idx               # [B]
        # last_eos_idx = torch.where(has_eos, last_eos_idx, torch.full_like(last_eos_idx, -1))
        # if self.debug:
        #     lens = attention_mask.long().sum(dim=1)  # [B]
        #     # last_idx는 lens-1이니까 같이 출력해도 됨
        #     print("T:", input_ids.size(1))
        #     print("lens:", lens.tolist())
        #     print("last_idx:", last_idx.tolist())
        #     print("picked_id:", picked_id.tolist())
        #     print("picked_is_eos:", (picked_id == self.eos_id).tolist())
        #     print("picked_is_pad:", (picked_id == self.pad_id).tolist())
        #     print("has_eos:", has_eos.tolist())
        #     print("last_eos_idx:", last_eos_idx.tolist())
        #     print("last_idx:", last_idx.tolist())
        #     print("last_idx_is_last_eos:", (last_eos_idx == last_idx).tolist())

        feats = []
        for li in self.use_layers:
            h = hs[li]  # [B,T,D]
            if self.pool == "mean":
                pooled = mean_pool(h, attention_mask)
            elif self.pool == "lastk":
                pooled = last_k_mean_pool(h, attention_mask, k=self.last_k)
            elif self.pool == "attn":
                pooled = self.attn_pool(h, attention_mask)
            else:  # "last"
                pooled = h[ar, last_idx, :]  # [B,D] 배치별로 last token을 추출 
            feats.append(pooled)

        feat = torch.cat(feats, dim=-1)  # 배치를 결합 
        logits = self.head(feat) # classifier에 넣고 돌림([batch size, hidden_states * # layers])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels.long()) # @ 수정 
            # if hasattr(self, "class_weight") and self.class_weight is not None:
            #     weight = self.class_weight.to(logits.device)
            #     loss = F.cross_entropy(logits.float(), labels.long(), weight=weight) # CE
            # else:
            #     loss = F.cross_entropy(logits.float(), labels.long())
        return {"loss": loss, "logits": logits}

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def _load_any(path: str):
    if path.endswith(".json") or path.endswith(".jsonl"):
        dsd = load_dataset("json", data_files=path)   # DatasetDict
    else:
        dsd = load_from_disk(path)                    # DatasetDict or Dataset
    if hasattr(dsd, "keys"):  # DatasetDict 처럼 동작
        if "train" in dsd:
            return dsd["train"]
        # train이 없으면 첫 split이라도
        return dsd[next(iter(dsd.keys()))]
    return dsd

def train(
    base_model: str = "gcyzsl/O3_LLAMA2_ScienceQA",
    data_path: str = "./data/scienceqa_SD_10/scienceqa_biology_writing-strategies_figurative-language_units-and-measurement_physics_reference-skills_chemistry_punctuation_economics_grammar_earth-science_train_SD.json",
    retain_data_path: str = "./data/scienceqa_RD_10/scienceqa_not_biology_writing-strategies_figurative-language_units-and-measurement_physics_reference-skills_chemistry_punctuation_economics_grammar_earth-science_train_RD.json",  
    output_dir: str = "./detecting_lora",
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 6,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512, # tokenizer max length 
    val_set_size: int = 0.2,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    load_in_8bit: bool = False,
    add_eos_token: bool = True,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = "", # None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # @ 수정 
    train_end_layer: int = 2, # 3개만 사용 
    pool = "mean"
    # proj_dim: int = 256,
    # m: float = 0.999,
    # tau: float = 0.07,
    # lambda_info: float = 0.1,
    # lambda_mlm: float = 0.1,   
):
    set_seed(seed)
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # 멀티 GPU인 경우 world_size가 1이 아님 
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    config = AutoConfig.from_pretrained(base_model)
    model_llm = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    N = int(train_end_layer)  
    model_llm.model.layers = nn.ModuleList(list(model_llm.model.layers[:N]))
    model_llm.config.num_hidden_layers = N
    model_llm.model.config.num_hidden_layers = N 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_layers = tuple(range(1, train_end_layer + 1))
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = EarlyLayerOODHead( # original 
        model_llm, 
        use_layers=use_layers,
        freeze_backbone=True,
        pool = pool,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        debug=True,   # 확인 끝나면 False로
    )
    
    ds_forget = _load_any(data_path).map(lambda x: {"cls_label": 1}) # ID
    ds_retain = _load_any(retain_data_path).map(lambda x: {"cls_label": 0}) # OOD
    full_ds = concatenate_datasets([ds_forget, ds_retain]).shuffle(seed=seed)
    full_ds = full_ds.cast_column("cls_label", ClassLabel(num_classes=2, names=["OOD", "ID"]))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100*trainable/total:.4f}")
        
    if val_set_size is None or val_set_size == 0:
        train_ds = full_ds
        val_ds = None
    else:
        test_size = float(val_set_size)
        train_val = full_ds.train_test_split( # default로 shuffle 
            test_size=test_size,
            seed=42,
            stratify_by_column="cls_label",
        )
        train_ds = train_val["train"]
        val_ds = train_val["test"]

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    def tokenize_one(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"], # output은 ID/OOD 탐지에 사용하지 x 
        )
        # tokenizer가 리턴하는 내용
        # {
        #     "input_ids": [...],          # 토큰 id들
        #     "attention_mask": [...],     # 1/0
        #     "labels": [...]              # 보통 input_ids 복사본
        # }
        tok = tokenizer(
            full_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )
        if (
            tok["input_ids"][-1] != tokenizer.eos_token_id
            and len(tok["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            tok["input_ids"].append(tokenizer.eos_token_id)
            tok["attention_mask"].append(1)
        tok["labels"] = int(data_point["cls_label"])
        return tok

    remove_cols = list(set(train_ds.column_names) - {"cls_label"})
    train_ds = train_ds.map(tokenize_one, remove_columns=remove_cols)

    if val_ds is not None:
        remove_cols_val = list(set(val_ds.column_names) - {"cls_label"})
        val_ds = val_ds.map(tokenize_one, remove_columns=remove_cols_val)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        # OOD(0)를 positive로(OOD detection이기 때문에)
        # sklearn은 positive=1을 기준으로 auc를 계산하므로
        y_true = (labels == 0).astype(np.int32)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        ood_score = probs[:, 0]  # OOD class prob
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(y_true, ood_score)
        except Exception as e:
            auroc = float("nan")
        return {"accuracy": acc, "auroc_ood": float(auroc)}
    
    # 확인용
    with torch.no_grad():
        tmp = tokenizer("hi", return_tensors="pt").to(next(model.parameters()).device)
        out0, o_loss = model.backbone(**tmp, output_hidden_states=True, return_dict=True, use_cache=False)
        print("out is  : ", out0)
        assert len(out0.hidden_states) >= (train_end_layer + 1)
    from collections import Counter
    print("train label dist:", Counter(train_ds["labels"]))
    print("val   label dist:", Counter(val_ds["labels"]))

    # train label dist: Counter({1: 4585, 0: 621})
    # val   label dist: Counter({1: 1147, 0: 155})
    cnt = Counter(train_ds["labels"])   # labels는 0/1 @ 수정 
    w0 = 1.0 / cnt[0]
    w1 = 1.0 / cnt[1]
    class_weight = torch.tensor([w0, w1], dtype=torch.float32)
    model.class_weight = class_weight
    print("class_weight:", class_weight)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics if val_ds is not None else None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer, 
            pad_to_multiple_of=8
        )
        # data_collator=default_data_collator
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)