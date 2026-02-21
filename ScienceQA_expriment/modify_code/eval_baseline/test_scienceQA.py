{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7b5d5267",
   "metadata": {},
   "source": [
    "# Llama 2 7B로 ScienceQA 평가 (IPython Notebook)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34e9aa59",
   "metadata": {},
   "outputs": [],
   "source": [
    "# (선택) 필요 패키지 설치\n",
    "# 환경에 이미 설치되어 있으면 건너뛰세요.\n",
    "\n",
    "%pip -q install -U transformers accelerate datasets evaluate sentencepiece\n",
    "\n",
    "# bitsandbytes(8bit/4bit 로딩) 사용하고 싶으면 아래도 (CUDA 환경에서만 권장)\n",
    "# %pip -q install -U bitsandbytes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e9e888df",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, json, re, math\n",
    "from dataclasses import dataclass\n",
    "from typing import Dict, Any, List, Optional, Tuple\n",
    "\n",
    "import torch\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0d9706cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.\n",
      "Token is valid (permission: read).\n",
      "Your token has been saved to /dataset/.cache/token\n",
      "Login successful\n"
     ]
    }
   ],
   "source": [
    "!huggingface-cli login --token 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"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c456df06",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ[\"HF_TOKEN\"] = \"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\"  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "7ff794b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0863314c47ed4853b9ef99a1fc04da6a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of LlamaForCausalLM were not initialized from the model checkpoint at meta-llama/Meta-Llama-3-8B-Instruct and are newly initialized: ['model.layers.20.self_attn.rotary_emb.inv_freq', 'model.layers.16.self_attn.rotary_emb.inv_freq', 'model.layers.1.self_attn.rotary_emb.inv_freq', 'model.layers.25.self_attn.rotary_emb.inv_freq', 'model.layers.11.self_attn.rotary_emb.inv_freq', 'model.layers.3.self_attn.rotary_emb.inv_freq', 'model.layers.29.self_attn.rotary_emb.inv_freq', 'model.layers.2.self_attn.rotary_emb.inv_freq', 'model.layers.12.self_attn.rotary_emb.inv_freq', 'model.layers.6.self_attn.rotary_emb.inv_freq', 'model.layers.0.self_attn.rotary_emb.inv_freq', 'model.layers.10.self_attn.rotary_emb.inv_freq', 'model.layers.18.self_attn.rotary_emb.inv_freq', 'model.layers.21.self_attn.rotary_emb.inv_freq', 'model.layers.24.self_attn.rotary_emb.inv_freq', 'model.layers.31.self_attn.rotary_emb.inv_freq', 'model.layers.8.self_attn.rotary_emb.inv_freq', 'model.layers.23.self_attn.rotary_emb.inv_freq', 'model.layers.9.self_attn.rotary_emb.inv_freq', 'model.layers.27.self_attn.rotary_emb.inv_freq', 'model.layers.13.self_attn.rotary_emb.inv_freq', 'model.layers.5.self_attn.rotary_emb.inv_freq', 'model.layers.15.self_attn.rotary_emb.inv_freq', 'model.layers.22.self_attn.rotary_emb.inv_freq', 'model.layers.14.self_attn.rotary_emb.inv_freq', 'model.layers.4.self_attn.rotary_emb.inv_freq', 'model.layers.30.self_attn.rotary_emb.inv_freq', 'model.layers.26.self_attn.rotary_emb.inv_freq', 'model.layers.28.self_attn.rotary_emb.inv_freq', 'model.layers.17.self_attn.rotary_emb.inv_freq', 'model.layers.7.self_attn.rotary_emb.inv_freq', 'model.layers.19.self_attn.rotary_emb.inv_freq']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    }
   ],
   "source": [
    "# Load model directly\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "import os\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\"\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(\n",
    "    \"meta-llama/Meta-Llama-3-8B-Instruct\", \n",
    "    token=os.environ[\"HF_TOKEN\"], \n",
    "    torch_dtype=torch.bfloat16,  \n",
    "    device_map={\"\": 0},      \n",
    ")\n",
    "\n",
    "model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Meta-Llama-3-8B-Instruct\", token=os.environ[\"HF_TOKEN\"],)\n",
    "messages = [\n",
    "    {\"role\": \"user\", \"content\": \"Who are you?\"},\n",
    "]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "41af59cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "device: cuda\n",
      "none\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LlamaForCausalLM(\n",
       "  (model): LlamaModel(\n",
       "    (embed_tokens): Embedding(128256, 4096, padding_idx=0)\n",
       "    (layers): ModuleList(\n",
       "      (0-31): 32 x LlamaDecoderLayer(\n",
       "        (self_attn): LlamaAttention(\n",
       "          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)\n",
       "          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)\n",
       "          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)\n",
       "          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)\n",
       "          (rotary_emb): LlamaRotaryEmbedding()\n",
       "        )\n",
       "        (mlp): LlamaMLP(\n",
       "          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)\n",
       "          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)\n",
       "          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)\n",
       "          (act_fn): SiLUActivation()\n",
       "        )\n",
       "        (input_layernorm): LlamaRMSNorm()\n",
       "        (post_attention_layernorm): LlamaRMSNorm()\n",
       "      )\n",
       "    )\n",
       "    (norm): LlamaRMSNorm()\n",
       "  )\n",
       "  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)\n",
       ")"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(\"device:\", device)\n",
    "if tokenizer.pad_token_id is None:\n",
    "    print('none')\n",
    "    tokenizer.pad_token = tokenizer.eos_token\n",
    "tokenizer.padding_side = \"left\"\n",
    "tokenizer.pad_token = tokenizer.eos_token  # llama 계열에서 안전\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "d21359bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "data = load_dataset('derek-thomas/ScienceQA', split='train') \n",
    "data_val = load_dataset('derek-thomas/ScienceQA', split='validation')\n",
    "data_test = load_dataset('derek-thomas/ScienceQA', split='test')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "c13104c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['image', 'question', 'choices', 'answer', 'hint', 'task', 'grade',\n",
       "       'subject', 'topic', 'category', 'skill', 'lecture', 'solution'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ds = data.to_pandas()\n",
    "ds.columns # question, hint, choices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "31ff4d27",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4241, 2224)"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from datasets import Dataset\n",
    "IMG_COL = \"image\"   # 필요하면 \"image_id\" / \"image_path\" 등으로 바꿔\n",
    "def is_no_image(x):\n",
    "    v = x.get(IMG_COL, None)\n",
    "    if v is None:\n",
    "        return True\n",
    "    if isinstance(v, str) and v.strip() == \"\":\n",
    "        return True\n",
    "    if isinstance(v, dict):\n",
    "        path = v.get(\"path\", None)\n",
    "        bytes_ = v.get(\"bytes\", None)\n",
    "        if (path is None or str(path).strip() == \"\") and (bytes_ is None):\n",
    "            return True\n",
    "    return False\n",
    "\n",
    "data_test_noimg = data_test.filter(is_no_image)\n",
    "data_val_noimg  = data_val.filter(is_no_image)\n",
    "data_train_noimg= data.filter(is_no_image)\n",
    "len(data_test), len(data_test_noimg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "b3b9c95a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('Below is an instruction that describes a task. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n Context: Hint: People who can knit had to learn how to do it.\\n Question: Is the following trait inherited or acquired?\\n# Sasha is good at knitting hats.\\n Options: (A) acquired (B) inherited\\n\\n\\n ### Response:\\nAnswer: The answer is C.',)"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"Below is an instruction that describes a task. \" \\\n",
    "\"Write a response that appropriately completes the request.\\n\\n \\\n",
    "### Instruction:\\n \\\n",
    "Context: Hint: People who can knit had to learn how to do it.\\n \\\n",
    "Question: Is the following trait inherited or acquired?\\n# Sasha is good at knitting hats.\\n \\\n",
    "Options: (A) acquired (B) inherited\\n\\n\\n \\\n",
    "### Response:\\nAnswer: The answer is C.\", "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "18ed1bd4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_cqm_instruction(context: str, question: str, opt: list[str]) -> str:\n",
    "    instr = (\n",
    "        f\"Context: {context}\\n\"\n",
    "        f\"Question: {question}\\n\"\n",
    "        f\"Options: {opt}\\n\"\n",
    "    )\n",
    "    return instr\n",
    "PROMPT_NO_INPUT = (\n",
    "    \"Below is an instruction that describes a task. \"\n",
    "    \"Write a response that appropriately completes the request.\\n\\n\"\n",
    "    \"### Instruction:\\n{instruction}\\n\\n### Response:\\n\"\n",
    ")\n",
    "def build_alpaca_prompt_cqm(context: str, question: str, choices: list[str]) -> str:\n",
    "    instruction = build_cqm_instruction(context, question, choices)\n",
    "    return PROMPT_NO_INPUT.format(instruction=instruction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "8de273e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_field(ex, keys, default=\"\"):\n",
    "    for k in keys:\n",
    "        if k in ex and ex[k] is not None:\n",
    "            return ex[k]\n",
    "    return default\n",
    "# question, hint, choices\n",
    "def example_to_prompt(ex):\n",
    "    question = get_field(ex, [\"question\"], \"\")\n",
    "    choices  = get_field(ex, [\"choices\"], \"ArithmeticError\")\n",
    "    context  = get_field(ex, [\"hint\"], \"\")\n",
    "    return build_alpaca_prompt_cqm(context, question, list(choices))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "340f291d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def parse_answer_letter(text: str) -> Optional[str]:\n",
    "    m = re.search(r\"\\b([A-E])\\b\", text.strip())\n",
    "    return m.group(1) if m else None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "3de68651",
   "metadata": {},
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def generate_answers(prompts: List[str], max_new_tokens: int = 16, temperature: float = 0.0) -> List[str]:\n",
    "    # temperature=0이면 greedy에 가깝게\n",
    "    inputs = tokenizer(prompts, return_tensors=\"pt\", padding=True, truncation=True, max_length=1024)\n",
    "    input_ids = inputs[\"input_ids\"].to(model.device)\n",
    "    attention_mask = inputs[\"attention_mask\"].to(model.device)\n",
    "\n",
    "    gen_cfg = GenerationConfig(\n",
    "        max_new_tokens=max_new_tokens,\n",
    "        do_sample=(temperature > 0),\n",
    "        temperature=temperature if temperature > 0 else None,\n",
    "        top_p=0.9 if temperature > 0 else None,\n",
    "        pad_token_id=tokenizer.pad_token_id,\n",
    "        eos_token_id=tokenizer.eos_token_id,\n",
    "    )\n",
    "    out = model.generate(\n",
    "        input_ids=input_ids,\n",
    "        attention_mask=attention_mask,\n",
    "        generation_config=gen_cfg,\n",
    "    )\n",
    "    generated = out[:, input_ids.shape[1]:]\n",
    "    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)\n",
    "    return texts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "1a77ee90",
   "metadata": {},
   "outputs": [],
   "source": [
    "CHOICE_LETTERS = [\"A\",\"B\",\"C\",\"D\",\"E\"]\n",
    "\n",
    "def get_gold_letter(ex):\n",
    "    g = ex.get(\"answer\", None)\n",
    "    if isinstance(g, int):\n",
    "        return CHOICE_LETTERS[g] if 0 <= g < 5 else None\n",
    "    if isinstance(g, str):\n",
    "        m = re.search(r\"[A-E]\", g)\n",
    "        return m.group(0) if m else None\n",
    "    return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "432b5f28",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tqdm import tqdm\n",
    "def evaluate_dataset(ds, batch_size=8, max_eval=None, temperature=0.0):\n",
    "    total = 0\n",
    "    correct = 0\n",
    "    skipped = 0\n",
    "    \n",
    "    prompts = []\n",
    "    golds = []\n",
    "\n",
    "    # select로 샘플 수 제한\n",
    "    it = ds\n",
    "    if max_eval is not None:\n",
    "        it = ds.select(range(min(len(ds), max_eval)))\n",
    "\n",
    "    for ex in tqdm(it, total=len(it)):\n",
    "        gold = get_gold_letter(ex)\n",
    "        if gold is None:\n",
    "            skipped += 1\n",
    "            continue\n",
    "\n",
    "        prompt = example_to_prompt(ex)   # 너가 만든 CQM->Alpaca prompt\n",
    "        prompts.append(prompt)\n",
    "        golds.append(gold)\n",
    "\n",
    "        if len(prompts) >= batch_size:\n",
    "            outs = generate_answers(prompts, max_new_tokens=16, temperature=temperature)\n",
    "            for out_text, gt in zip(outs, golds):\n",
    "                pred = parse_answer_letter(out_text)\n",
    "                total += 1\n",
    "                correct += int(pred == gt)\n",
    "            prompts, golds = [], []\n",
    "\n",
    "    # 남은 배치 처리\n",
    "    if prompts:\n",
    "        outs = generate_answers(prompts, max_new_tokens=16, temperature=temperature)\n",
    "        for out_text, gt in zip(outs, golds):\n",
    "            pred = parse_answer_letter(out_text)\n",
    "            total += 1\n",
    "            correct += int(pred == gt)\n",
    "\n",
    "    acc = correct / total if total > 0 else float(\"nan\")\n",
    "    return {\n",
    "        \"total_scored\": total,\n",
    "        \"correct\": correct,\n",
    "        \"accuracy\": acc,\n",
    "        \"skipped_no_gold\": skipped,\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "7ba8b1fc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch: 2.2.1+cu121\n",
      "cuda available: True\n",
      "device count: 4\n",
      "current device: 0\n",
      "gpu name: NVIDIA H100 80GB HBM3\n",
      "model device: cpu\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "print(\"torch:\", torch.__version__)\n",
    "print(\"cuda available:\", torch.cuda.is_available())\n",
    "print(\"device count:\", torch.cuda.device_count())\n",
    "if torch.cuda.is_available():\n",
    "    print(\"current device:\", torch.cuda.current_device())\n",
    "    print(\"gpu name:\", torch.cuda.get_device_name(0))\n",
    "import torch\n",
    "print(\"model device:\", next(model.parameters()).device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "ec08dd7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  7%|▋         | 143/2144 [05:47<1:21:06,  2.43s/it]\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[52], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m val_res \u001b[38;5;241m=\u001b[39m \u001b[43mevaluate_dataset\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdata_val_noimg\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m8\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmax_eval\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtemperature\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m0.0\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m      2\u001b[0m test_res \u001b[38;5;241m=\u001b[39m evaluate_dataset(data_test_noimg, batch_size\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m8\u001b[39m, max_eval\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, temperature\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.0\u001b[39m)\n\u001b[1;32m      4\u001b[0m val_res, test_res\n",
      "Cell \u001b[0;32mIn[51], line 26\u001b[0m, in \u001b[0;36mevaluate_dataset\u001b[0;34m(ds, batch_size, max_eval, temperature)\u001b[0m\n\u001b[1;32m     23\u001b[0m golds\u001b[38;5;241m.\u001b[39mappend(gold)\n\u001b[1;32m     25\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(prompts) \u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m batch_size:\n\u001b[0;32m---> 26\u001b[0m     outs \u001b[38;5;241m=\u001b[39m \u001b[43mgenerate_answers\u001b[49m\u001b[43m(\u001b[49m\u001b[43mprompts\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmax_new_tokens\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m16\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtemperature\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtemperature\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     27\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m out_text, gt \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mzip\u001b[39m(outs, golds):\n\u001b[1;32m     28\u001b[0m         pred \u001b[38;5;241m=\u001b[39m parse_answer_letter(out_text)\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/utils/_contextlib.py:115\u001b[0m, in \u001b[0;36mcontext_decorator.<locals>.decorate_context\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    112\u001b[0m \u001b[38;5;129m@functools\u001b[39m\u001b[38;5;241m.\u001b[39mwraps(func)\n\u001b[1;32m    113\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21mdecorate_context\u001b[39m(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m    114\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m ctx_factory():\n\u001b[0;32m--> 115\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[0;32mIn[49], line 16\u001b[0m, in \u001b[0;36mgenerate_answers\u001b[0;34m(prompts, max_new_tokens, temperature)\u001b[0m\n\u001b[1;32m      6\u001b[0m attention_mask \u001b[38;5;241m=\u001b[39m inputs[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mattention_mask\u001b[39m\u001b[38;5;124m\"\u001b[39m]\u001b[38;5;241m.\u001b[39mto(model\u001b[38;5;241m.\u001b[39mdevice)\n\u001b[1;32m      8\u001b[0m gen_cfg \u001b[38;5;241m=\u001b[39m GenerationConfig(\n\u001b[1;32m      9\u001b[0m     max_new_tokens\u001b[38;5;241m=\u001b[39mmax_new_tokens,\n\u001b[1;32m     10\u001b[0m     do_sample\u001b[38;5;241m=\u001b[39m(temperature \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m0\u001b[39m),\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     14\u001b[0m     eos_token_id\u001b[38;5;241m=\u001b[39mtokenizer\u001b[38;5;241m.\u001b[39meos_token_id,\n\u001b[1;32m     15\u001b[0m )\n\u001b[0;32m---> 16\u001b[0m out \u001b[38;5;241m=\u001b[39m \u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgenerate\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m     17\u001b[0m \u001b[43m    \u001b[49m\u001b[43minput_ids\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minput_ids\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     18\u001b[0m \u001b[43m    \u001b[49m\u001b[43mattention_mask\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mattention_mask\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     19\u001b[0m \u001b[43m    \u001b[49m\u001b[43mgeneration_config\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgen_cfg\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     20\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     21\u001b[0m generated \u001b[38;5;241m=\u001b[39m out[:, input_ids\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m1\u001b[39m]:]\n\u001b[1;32m     22\u001b[0m texts \u001b[38;5;241m=\u001b[39m tokenizer\u001b[38;5;241m.\u001b[39mbatch_decode(generated, skip_special_tokens\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/utils/_contextlib.py:115\u001b[0m, in \u001b[0;36mcontext_decorator.<locals>.decorate_context\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    112\u001b[0m \u001b[38;5;129m@functools\u001b[39m\u001b[38;5;241m.\u001b[39mwraps(func)\n\u001b[1;32m    113\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21mdecorate_context\u001b[39m(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m    114\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m ctx_factory():\n\u001b[0;32m--> 115\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/generation/utils.py:1538\u001b[0m, in \u001b[0;36mGenerationMixin.generate\u001b[0;34m(self, inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, **kwargs)\u001b[0m\n\u001b[1;32m   1532\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[1;32m   1533\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnum_return_sequences has to be 1 when doing greedy search, \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   1534\u001b[0m             \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbut is \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mgeneration_config\u001b[38;5;241m.\u001b[39mnum_return_sequences\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   1535\u001b[0m         )\n\u001b[1;32m   1537\u001b[0m     \u001b[38;5;66;03m# 11. run greedy search\u001b[39;00m\n\u001b[0;32m-> 1538\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgreedy_search\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1539\u001b[0m \u001b[43m        \u001b[49m\u001b[43minput_ids\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1540\u001b[0m \u001b[43m        \u001b[49m\u001b[43mlogits_processor\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlogits_processor\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1541\u001b[0m \u001b[43m        \u001b[49m\u001b[43mstopping_criteria\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mstopping_criteria\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1542\u001b[0m \u001b[43m        \u001b[49m\u001b[43mpad_token_id\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgeneration_config\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mpad_token_id\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1543\u001b[0m \u001b[43m        \u001b[49m\u001b[43meos_token_id\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgeneration_config\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43meos_token_id\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1544\u001b[0m \u001b[43m        \u001b[49m\u001b[43moutput_scores\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgeneration_config\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moutput_scores\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1545\u001b[0m \u001b[43m        \u001b[49m\u001b[43mreturn_dict_in_generate\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgeneration_config\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mreturn_dict_in_generate\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1546\u001b[0m \u001b[43m        \u001b[49m\u001b[43msynced_gpus\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msynced_gpus\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1547\u001b[0m \u001b[43m        \u001b[49m\u001b[43mstreamer\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mstreamer\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1548\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mmodel_kwargs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1549\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1551\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m is_contrastive_search_gen_mode:\n\u001b[1;32m   1552\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m generation_config\u001b[38;5;241m.\u001b[39mnum_return_sequences \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/generation/utils.py:2362\u001b[0m, in \u001b[0;36mGenerationMixin.greedy_search\u001b[0;34m(self, input_ids, logits_processor, stopping_criteria, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus, streamer, **model_kwargs)\u001b[0m\n\u001b[1;32m   2359\u001b[0m model_inputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mprepare_inputs_for_generation(input_ids, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mmodel_kwargs)\n\u001b[1;32m   2361\u001b[0m \u001b[38;5;66;03m# forward pass to get next token\u001b[39;00m\n\u001b[0;32m-> 2362\u001b[0m outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m(\u001b[49m\n\u001b[1;32m   2363\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mmodel_inputs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   2364\u001b[0m \u001b[43m    \u001b[49m\u001b[43mreturn_dict\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m   2365\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_attentions\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43moutput_attentions\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   2366\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_hidden_states\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43moutput_hidden_states\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   2367\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   2369\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m synced_gpus \u001b[38;5;129;01mand\u001b[39;00m this_peer_finished:\n\u001b[1;32m   2370\u001b[0m     \u001b[38;5;28;01mcontinue\u001b[39;00m  \u001b[38;5;66;03m# don't waste resources running the code we don't need\u001b[39;00m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[1;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[1;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[1;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[1;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[1;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[0;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:806\u001b[0m, in \u001b[0;36mLlamaForCausalLM.forward\u001b[0;34m(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)\u001b[0m\n\u001b[1;32m    803\u001b[0m return_dict \u001b[38;5;241m=\u001b[39m return_dict \u001b[38;5;28;01mif\u001b[39;00m return_dict \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39muse_return_dict\n\u001b[1;32m    805\u001b[0m \u001b[38;5;66;03m# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)\u001b[39;00m\n\u001b[0;32m--> 806\u001b[0m outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmodel\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    807\u001b[0m \u001b[43m    \u001b[49m\u001b[43minput_ids\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minput_ids\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    808\u001b[0m \u001b[43m    \u001b[49m\u001b[43mattention_mask\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mattention_mask\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    809\u001b[0m \u001b[43m    \u001b[49m\u001b[43mposition_ids\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mposition_ids\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    810\u001b[0m \u001b[43m    \u001b[49m\u001b[43mpast_key_values\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mpast_key_values\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    811\u001b[0m \u001b[43m    \u001b[49m\u001b[43minputs_embeds\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minputs_embeds\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    812\u001b[0m \u001b[43m    \u001b[49m\u001b[43muse_cache\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43muse_cache\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    813\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_attentions\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43moutput_attentions\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    814\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_hidden_states\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43moutput_hidden_states\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    815\u001b[0m \u001b[43m    \u001b[49m\u001b[43mreturn_dict\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mreturn_dict\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    816\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    818\u001b[0m hidden_states \u001b[38;5;241m=\u001b[39m outputs[\u001b[38;5;241m0\u001b[39m]\n\u001b[1;32m    819\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpretraining_tp \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[1;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[1;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[1;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[1;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[1;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[0;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:693\u001b[0m, in \u001b[0;36mLlamaModel.forward\u001b[0;34m(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)\u001b[0m\n\u001b[1;32m    685\u001b[0m     layer_outputs \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mutils\u001b[38;5;241m.\u001b[39mcheckpoint\u001b[38;5;241m.\u001b[39mcheckpoint(\n\u001b[1;32m    686\u001b[0m         create_custom_forward(decoder_layer),\n\u001b[1;32m    687\u001b[0m         hidden_states,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    690\u001b[0m         \u001b[38;5;28;01mNone\u001b[39;00m,\n\u001b[1;32m    691\u001b[0m     )\n\u001b[1;32m    692\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--> 693\u001b[0m     layer_outputs \u001b[38;5;241m=\u001b[39m \u001b[43mdecoder_layer\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    694\u001b[0m \u001b[43m        \u001b[49m\u001b[43mhidden_states\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    695\u001b[0m \u001b[43m        \u001b[49m\u001b[43mattention_mask\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mattention_mask\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    696\u001b[0m \u001b[43m        \u001b[49m\u001b[43mposition_ids\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mposition_ids\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    697\u001b[0m \u001b[43m        \u001b[49m\u001b[43mpast_key_value\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mpast_key_value\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    698\u001b[0m \u001b[43m        \u001b[49m\u001b[43moutput_attentions\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43moutput_attentions\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    699\u001b[0m \u001b[43m        \u001b[49m\u001b[43muse_cache\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43muse_cache\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    700\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    702\u001b[0m hidden_states \u001b[38;5;241m=\u001b[39m layer_outputs[\u001b[38;5;241m0\u001b[39m]\n\u001b[1;32m    704\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m use_cache:\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[1;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[1;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[1;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[1;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[1;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[0;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:408\u001b[0m, in \u001b[0;36mLlamaDecoderLayer.forward\u001b[0;34m(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)\u001b[0m\n\u001b[1;32m    405\u001b[0m hidden_states \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39minput_layernorm(hidden_states)\n\u001b[1;32m    407\u001b[0m \u001b[38;5;66;03m# Self Attention\u001b[39;00m\n\u001b[0;32m--> 408\u001b[0m hidden_states, self_attn_weights, present_key_value \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mself_attn\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    409\u001b[0m \u001b[43m    \u001b[49m\u001b[43mhidden_states\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mhidden_states\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    410\u001b[0m \u001b[43m    \u001b[49m\u001b[43mattention_mask\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mattention_mask\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    411\u001b[0m \u001b[43m    \u001b[49m\u001b[43mposition_ids\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mposition_ids\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    412\u001b[0m \u001b[43m    \u001b[49m\u001b[43mpast_key_value\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mpast_key_value\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    413\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_attentions\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43moutput_attentions\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    414\u001b[0m \u001b[43m    \u001b[49m\u001b[43muse_cache\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43muse_cache\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    415\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    416\u001b[0m hidden_states \u001b[38;5;241m=\u001b[39m residual \u001b[38;5;241m+\u001b[39m hidden_states\n\u001b[1;32m    418\u001b[0m \u001b[38;5;66;03m# Fully Connected\u001b[39;00m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1511\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1509\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[1;32m   1510\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 1511\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_impl\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/torch/nn/modules/module.py:1520\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1515\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[1;32m   1516\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[1;32m   1517\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[1;32m   1518\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[1;32m   1519\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[0;32m-> 1520\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mforward_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1522\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m   1523\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:327\u001b[0m, in \u001b[0;36mLlamaAttention.forward\u001b[0;34m(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)\u001b[0m\n\u001b[1;32m    324\u001b[0m past_key_value \u001b[38;5;241m=\u001b[39m (key_states, value_states) \u001b[38;5;28;01mif\u001b[39;00m use_cache \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    326\u001b[0m \u001b[38;5;66;03m# repeat k/v heads if n_kv_heads < n_heads\u001b[39;00m\n\u001b[0;32m--> 327\u001b[0m key_states \u001b[38;5;241m=\u001b[39m \u001b[43mrepeat_kv\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey_states\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mnum_key_value_groups\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    328\u001b[0m value_states \u001b[38;5;241m=\u001b[39m repeat_kv(value_states, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mnum_key_value_groups)\n\u001b[1;32m    330\u001b[0m attn_weights \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mmatmul(query_states, key_states\u001b[38;5;241m.\u001b[39mtranspose(\u001b[38;5;241m2\u001b[39m, \u001b[38;5;241m3\u001b[39m)) \u001b[38;5;241m/\u001b[39m math\u001b[38;5;241m.\u001b[39msqrt(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhead_dim)\n",
      "File \u001b[0;32m/dataset/dataset/envs/hai_ssh_o3/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:230\u001b[0m, in \u001b[0;36mrepeat_kv\u001b[0;34m(hidden_states, n_rep)\u001b[0m\n\u001b[1;32m    228\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m hidden_states\n\u001b[1;32m    229\u001b[0m hidden_states \u001b[38;5;241m=\u001b[39m hidden_states[:, :, \u001b[38;5;28;01mNone\u001b[39;00m, :, :]\u001b[38;5;241m.\u001b[39mexpand(batch, num_key_value_heads, n_rep, slen, head_dim)\n\u001b[0;32m--> 230\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mhidden_states\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mreshape\u001b[49m\u001b[43m(\u001b[49m\u001b[43mbatch\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnum_key_value_heads\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43m \u001b[49m\u001b[43mn_rep\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mslen\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mhead_dim\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "val_res = evaluate_dataset(data_val_noimg, batch_size=8, max_eval=None, temperature=0.0)\n",
    "test_res = evaluate_dataset(data_test_noimg, batch_size=8, max_eval=None, temperature=0.0)\n",
    "\n",
    "val_res, test_res"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c5437a4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "created": "2026-01-18T14:04:08.898655",
  "kernelspec": {
   "display_name": "hai_ssh_o3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
