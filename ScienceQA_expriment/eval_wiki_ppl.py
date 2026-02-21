# eval_wiki_ppl_base_only.py
# 목적: LoRA를 "전혀" 붙이지 않고(base only) WikiText-2 perplexity(PPL) 측정
#
# 실행 예)
#   python eval_wiki_ppl_base_only.py \
#     --base_model gcyzsl/O3_LLAMA2_ScienceQA \
#     --block_size 1024 --stride 1024 --split test
#
# 주의:
# - LoRA 관련(PeftModel/get_peft_model) 호출은 "절대" 하지 않음 = base only
# - 너 코드베이스의 커스텀 모델 LlamaForCausalLM_ood를 그대로 사용

import os
import math
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, LlamaTokenizer

from src.modeling_llama_hacked_o import LlamaForCausalLM_ood

os.environ["WANDB_DISABLED"] = "true"


@torch.no_grad()
def wikitext_ppl(
    model,
    tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    block_size=1024,
    stride=1024,
    device="cuda",
):
    """
    표준 LM PPL:
      - WikiText 텍스트를 이어붙여 토크나이즈
      - chunk 단위로 labels=input_ids로 loss 계산
      - NLL/token 평균 -> exp
    """
    model.eval()
    model.to(device)

    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]  # [T]

    nll_sum = 0.0
    tok_count = 0

    # non-overlap가 기본(stride=block_size). 필요 시 stride를 줄여 overlap 가능.
    for start in tqdm(range(0, input_ids.size(0) - 1, stride), desc="eval"):
        end = min(start + block_size, input_ids.size(0))
        chunk = input_ids[start:end]
        if chunk.numel() < 2:
            continue

        chunk = chunk.unsqueeze(0).to(device)  # [1, L]
        out = model(chunk, labels=chunk)        # HF causal LM loss(shift 포함 평균)

        n_tokens = chunk.size(1) - 1            # shift 때문에 L-1 토큰에 대해 loss
        nll_sum += out.loss.item() * n_tokens
        tok_count += n_tokens

        if end == input_ids.size(0):
            break

    ppl = math.exp(nll_sum / max(tok_count, 1))
    return ppl


def load_base_only(
    base_model: str,
    device_map="auto",
    torch_dtype=torch.bfloat16,
):
    config = AutoConfig.from_pretrained(base_model)

    # 커스텀 모델이 기대하는 config 필드 기본값 보장
    if not hasattr(config, "orthogonal_loss"):
        config.orthogonal_loss = False
    if not hasattr(config, "orthogonal_loss_weight"):
        config.orthogonal_loss_weight = 0.0
    if not hasattr(config, "lora_target_modules"):
        config.lora_target_modules = []

    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # ✅ forward에서 참조하는 attribute들을 "반드시" 만들어줌 (base-only라도 필요)
    # orthogonal loss 비활성 + olora_weights 비어있게
    if hasattr(model, "init_olora"):
        model.init_olora(orthogonal_loss=False, olora_weights=[])
    else:
        # 혹시 init_olora가 없으면 최소한 attribute만 만들어줌
        model.olora_weights = {}

    if hasattr(model, "init_oodweight"):
        model.init_oodweight(ood_weight=[1, 1])  # forward에서 쓰면 기본값
    else:
        model.ood_weight = [1, 1]

    if hasattr(model, "init_active_adapters_d"):
        model.init_active_adapters_d(active_adapters_d=["default"])

    if hasattr(model, "init_lora_start_layer"):
        model.init_lora_start_layer(lora_start_layer=0)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    return model, tokenizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="예: gcyzsl/O3_LLAMA2_ScienceQA 또는 로컬 경로")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu (cuda 사용 가능하면 cuda 추천)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # ✅ base only 로드 (LoRA attach 절대 안 함)
    model, tokenizer = load_base_only(
        base_model=args.base_model,
        device_map="auto" if args.device == "cuda" else None,
        torch_dtype=torch_dtype,
    )

    ppl = wikitext_ppl(
        model=model,
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split=args.split,
        block_size=args.block_size,
        stride=args.stride,
        device=args.device,
    )

    print(f"[BASE ONLY] base_model={args.base_model}")
    print(f"[WikiText-2 raw | split={args.split}] PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()
