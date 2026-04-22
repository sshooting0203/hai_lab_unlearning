from transformers import AutoTokenizer

def get_eoi_tokens(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Method 1: Using apply_chat_template (more robust for Instruct models)
    messages = [{"role": "user", "content": "DUMMY_INSTRUCTION"}]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    if "DUMMY_INSTRUCTION" in full_prompt:
        prefix, suffix = full_prompt.split("DUMMY_INSTRUCTION")
        eoi_ids = tokenizer.encode(suffix, add_special_tokens=False)
        print(f"\nModel: {model_id}")
        print(f"Full prompt template (with DUMMY_INSTRUCTION):\n{repr(full_prompt)}")
        print(f"EOI Suffix: {repr(suffix)}")
        print(f"EOI Token IDs: {eoi_ids}")
        print(f"EOI Tokens decoded: {[tokenizer.decode([tid]) for tid in eoi_ids]}")
    else:
        print(f"Could not find DUMMY_INSTRUCTION in prompt for {model_id}")

# Check Llama 3
try:
    get_eoi_tokens("meta-llama/Meta-Llama-3-8B-Instruct")
except Exception as e:
    print(f"Error loading Llama 3: {e}")

# Check Llama 3.1
try:
    get_eoi_tokens("meta-llama/Meta-Llama-3.1-8B-Instruct")
except Exception as e:
    print(f"Error loading Llama 3.1: {e}")

# Model: meta-llama/Meta-Llama-3-8B-Instruct
# Full prompt template (with DUMMY_INSTRUCTION):
# '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDUMMY_INSTRUCTION<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
# EOI Suffix: '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
# EOI Token IDs: [128009, 128006, 78191, 128007, 271]
# EOI Tokens decoded: ['<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n']

# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Full prompt template (with DUMMY_INSTRUCTION):
# '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nDUMMY_INSTRUCTION<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
# EOI Suffix: '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
# EOI Token IDs: [128009, 128006, 78191, 128007, 271]
# EOI Tokens decoded: ['<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n']