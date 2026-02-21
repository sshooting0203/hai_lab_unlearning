from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    use_fast=True
)

print("A:", tokenizer("A", add_special_tokens=False).input_ids)
print(" A:", tokenizer(" A", add_special_tokens=False).input_ids)
print("The answer is A:", tokenizer("The answer is A", add_special_tokens=False).input_ids)
print("The answer is", tokenizer("The answer is", add_special_tokens=False).input_ids)
