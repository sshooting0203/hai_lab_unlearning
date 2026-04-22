# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Question: What information supports the conclusion that Keith acquired this trait?\nOptions: (A) Keith likes to look at butterflies and beetles. (B) Keith learned to identify insects by reading many books about insects.\n\n\n### Response:",

# content_message="""
# Question: A supermarket is uncommon in what type of collection of shops?\n
# Options: 
# (A) strip mall 
# (B) city or town 
# (C) shoppingcentre 
# (D) boutique 
# (E) vermont\n\n\
# Answer with a single letter :
# """
# content_message="""
# Question: What information supports the conclusion that Keith acquired this trait?\n
# Options: 
# (A) Keith likes to look at butterflies and beetles. 
# (B) Keith learned to identify insects by reading many books about insects.\n\n\n
# Answer with a single letter :
# """
# content_message="""
# Question: Which is recyclable?\n
# Options: 
# (A) An Elephant 
# (B) A school notebook 
# (C) A boat 
# (D) A lake\n\n\n
# Answer with a single letter :
# """                    
# content_message="""
# Question: Where water be located in its gas form?\n
# Options: 
# (A) inside a disc golf driver 
# (B) inside of a brass pipe 
# (C) a mile up in the sky 
# (D) inside a leather baseball\n\n\n
# Answer with a single letter :
# """

# content_message="""
# Decide whether the hypothesis is entailed by the premise. 
# Return only one label: entailment or not_entailment.\n\n
# Premise: No Weapons of Mass Destruction Found in Iraq Yet.\n
# Hypothesis: Weapons of Mass Destruction Found in Iraq.\n\n
# """

content_message="""
Classify the user's intent from the following utterance.
Return only the intent label.\n\n
Utterance: help me change my oil\n
Intent: 
"""
messages = [
    {"role": "user", "content": f"{content_message}"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

