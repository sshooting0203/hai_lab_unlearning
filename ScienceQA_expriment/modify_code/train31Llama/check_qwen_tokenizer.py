from transformers import AutoTokenizer

model_id = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Hello"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Template output:\n{repr(prompt)}")

encoded = tokenizer(prompt, add_special_tokens=False)
print(f"Encoded with add_special_tokens=False: {encoded.input_ids}")

encoded_with_special = tokenizer(prompt, add_special_tokens=True)
print(f"Encoded with add_special_tokens=True: {encoded_with_special.input_ids}")

print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Qwen/Qwen2-7B-Instruct
# config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 663/663 [00:00<00:00, 4.36MB/s]
# tokenizer_config.json: 1.29kB [00:00, 5.61MB/s]
# vocab.json: 2.78MB [00:00, 11.7MB/s]
# merges.txt: 1.67MB [00:00, 11.0MB/s]
# tokenizer.json: 7.03MB [00:00, 17.8MB/s]
# Template output:
# '<|im_start|>system\n
# You are a helpful assistant.<|im_end|>\n
# <|im_start|>user\n
# Hello<|im_end|>\n
# <|im_start|>assistant\n'
# Encoded with add_special_tokens=False: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# Encoded with add_special_tokens=True: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# EOS token: <|im_end|> (ID: 151645)
# PAD token: <|endoftext|> (ID: 151643)
# Template output:
# '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'
# Encoded with add_special_tokens=False: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# Encoded with add_special_tokens=True: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# EOS token: <|im_end|> (ID: 151645)
# PAD token: <|endoftext|> (ID: 151643)

# Qwen/Qwen2.5-7B-Instruct
# config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 663/663 [00:00<00:00, 3.59MB/s]
# tokenizer_config.json: 7.30kB [00:00, 28.6MB/s]
# vocab.json: 2.78MB [00:00, 6.23MB/s]
# merges.txt: 1.67MB [00:00, 11.7MB/s]
# tokenizer.json: 7.03MB [00:00, 17.5MB/s]
# Template output:
# '<|im_start|>system\n
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
# <|im_end|>\n
# <|im_start|>user\n
# Hello<|im_end|>\n
# <|im_start|>assistant\n'
# Encoded with add_special_tokens=False: [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# Encoded with add_special_tokens=True: [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# EOS token: <|im_end|> (ID: 151645)
# PAD token: <|endoftext|> (ID: 151643)

# Template output:
# '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'
# Encoded with add_special_tokens=False: [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# Encoded with add_special_tokens=True: [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
# EOS token: <|im_end|> (ID: 151645)
# PAD token: <|endoftext|> (ID: 151643)