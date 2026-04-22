from transformers import AutoTokenizer

def get_eoi_tokens(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Qwen chat template used in qwen2_model.py and qwen25model.py
    QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    # Method 1: Using the split method from the code
    suffix = QWEN_CHAT_TEMPLATE.split("{instruction}")[-1]
    eoi_ids_manual = tokenizer.encode(suffix, add_special_tokens=False)
    
    # Method 2: Using apply_chat_template (more robust for Instruct models)
    messages = [{"role": "user", "content": "DUMMY_INSTRUCTION"}]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    suffix_auto = ""
    eoi_ids_auto = []
    if "DUMMY_INSTRUCTION" in full_prompt:
        suffix_auto = full_prompt.split("DUMMY_INSTRUCTION")[-1]
        eoi_ids_auto = tokenizer.encode(suffix_auto, add_special_tokens=False)
    
    print(f"\nModel: {model_id}")
    print(f"Manual Suffix (from template): {repr(suffix)}")
    print(f"Manual EOI Token IDs: {eoi_ids_manual}")
    print(f"Manual EOI Tokens decoded: {[tokenizer.decode([tid]) for tid in eoi_ids_manual]}")
    
    print(f"Auto Suffix (from apply_chat_template): {repr(suffix_auto)}")
    print(f"Auto EOI Token IDs: {eoi_ids_auto}")
    print(f"Auto EOI Tokens decoded: {[tokenizer.decode([tid]) for tid in eoi_ids_auto]}")

# Check Qwen2
try:
    get_eoi_tokens("Qwen/Qwen2-7B-Instruct")
except Exception as e:
    print(f"Error loading Qwen2: {e}")

# Check Qwen2.5
try:
    get_eoi_tokens("Qwen/Qwen2.5-7B-Instruct")
except Exception as e:
    print(f"Error loading Qwen2.5: {e}")

# Model: Qwen/Qwen2-7B-Instruct
# Manual Suffix (from template): '<|im_end|>\n<|im_start|>assistant\n'
# Manual EOI Token IDs: [151645, 198, 151644, 77091, 198]
# Manual EOI Tokens decoded: ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
# Auto Suffix (from apply_chat_template): '<|im_end|>\n<|im_start|>assistant\n'
# Auto EOI Token IDs: [151645, 198, 151644, 77091, 198]
# Auto EOI Tokens decoded: ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']

# Model: Qwen/Qwen2.5-7B-Instruct
# Manual Suffix (from template): '<|im_end|>\n<|im_start|>assistant\n'
# Manual EOI Token IDs: [151645, 198, 151644, 77091, 198]
# Manual EOI Tokens decoded: ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
# Auto Suffix (from apply_chat_template): '<|im_end|>\n<|im_start|>assistant\n'
# Auto EOI Token IDs: [151645, 198, 151644, 77091, 198]
# Auto EOI Tokens decoded: ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
