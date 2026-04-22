# Copyright (c) Meta Platforms, Inc. and affiliates.
import json, torch, datasets
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from datasets import Dataset as HFDataset
from src_lunar.model_utils.llama31model import build_prompt_from_instruction

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA2_CHAT_TEMPLATE = "[INST] {instruction} [/INST]" # @@@ : convert_raw_data_to_model_qa, convert_raw_questions_to_model_questions

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

ZEPHYR_CHAT_TEMPLATE = """
<|user|> {instruction} <|assistant|>
"""

##################################################################
def convert_raw_data_to_model_qa(tokenizer, max_length, question, answer, configs): # @@@ : eval할 때 input_ids, labels, attention_mask 리턴함 
    if configs.model_family == "llama3.1-8b-instruct":
        return convert_raw_data_to_llama31_qa(tokenizer, max_length, question, answer, configs)
    elif configs.model_family == "llama3-8b-instruct":
        new_question = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "Qwen2-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "Qwen2.5-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "llama2-7b-chat":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "gemma-7b-it":
        new_question = GEMMA_CHAT_TEMPLATE.format(instruction=question)
    else:
        raise ValueError(f"Invalid model_family")

    full_text = new_question + answer # answer를 text_input 뒤에 붙이기
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    tokenizer.padding_side = "left"
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)

    pad_input_ids = encoded["input_ids"] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded["attention_mask"] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = (
            encoded["input_ids"] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        )
    # change label to -100 for question tokens
    for i in range(num_question_tokens):
        label[i] = -100
    return (
        torch.tensor(pad_input_ids),
        torch.tensor(label),
        torch.tensor(pad_attention_mask),
    )

# llama3.1-8b-instruct
def convert_raw_data_to_llama31_qa(tokenizer, max_length, question, answer, configs):
    prompt_text = build_prompt_from_instruction(tokenizer, question, system=None)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_len = max_length - len(input_ids)

    input_ids += [pad_id] * pad_len
    attention_mask += [0] * pad_len
    labels += [-100] * pad_len

    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attention_mask),
    )

def convert_raw_questions_to_model_questions(tokenizer, max_length, question, configs):
    # question_start_token =  configs.question_start_tag, question_end_token = configs.question_end_tag
    if isinstance(question, list):
        question = str(question)  # Join list into a string

    if configs.model_family == "llama3-8b-instruct":
        new_question = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "Qwen2-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "Qwen2.5-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "llama2-7b-chat":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "zephyr-7b":
        new_question = ZEPHYR_CHAT_TEMPLATE.format(instruction=question)
    elif configs.model_family == "gemma-7b-it":
        new_question = GEMMA_CHAT_TEMPLATE.format(instruction=question)
    else:
        raise ValueError(f"Invalid model_family")

    # Tokenize the question
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        new_question,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # Ensure the output is padded to max_length
        return_tensors="pt",  # Return tensors directly
    )

    # Extract input_ids and attention_mask
    pad_input_ids = encoded["input_ids"].squeeze()  # Remove extra batch dimension
    pad_attention_mask = encoded[
        "attention_mask"
    ].squeeze()  # Remove extra batch dimension
    return pad_input_ids, pad_attention_mask

##################################################################
def dataset_format_qa(input_QA_list):
    all_QA_dict = {}
    for item in input_QA_list:
        edge = item["edge"]
        # Remove the "edge" key for the new format
        item_dict = {"question": item["question"], "answer": item["answer"]}
        # Check if the edge key exists in the result dictionary
        if edge not in all_QA_dict:
            all_QA_dict[edge] = []
        # Append the item dictionary to the corresponding edge list
        all_QA_dict[edge].append(item_dict)
    return all_QA_dict

def dataset_format_converstion(data_path, key="forget"):
    with open(data_path, "r") as f:
        all_QA_list = json.load(f)
    qa_list = []
    for item in all_QA_list:
        topic = item.get("topic") or item.get("domain")
        qa_list.append({
            "question": item["text_input"], # @@@ : 수정
            "answer": item["answer"],  
            "topic": topic,
        })
    return qa_list

def dataset_format_converstion_tofu(data_path, key="forget"):
    with open(data_path, "r") as f:
        all_QA_list = json.load(f)
    qa_list = []
    for item in all_QA_list:
        topic = item.get("topic") or item.get("domain")
        qa_list.append({
            "question": item.get("question") or item.get("text_input"),
            "answer": item.get("answer"),
            "paraphrased_answer": item.get("paraphrased_answer", ""),
            "perturbed_answer": item.get("perturbed_answer", []),
            "paraphrased_question": item.get("paraphrased_question", ""),
            "author_idx": item.get("author_idx"),
            "forget_author": item.get("forget_author"),
            "topic": topic,
        })
    return qa_list
#================QuestionsDataset===============
class QuestionsDataset(Dataset):
    def __init__(
        self,
        input_QA_list,
        tokenizer,
        configs,
        max_length=512,
        question_key="question",
        answer_key="answer",
    ):
        super(QuestionsDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_qa(input_QA_list)
        all_QA_list = []
        for sublist in all_QA.values():
            all_QA_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict = {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in all_QA_list[0]:
            QA_dict[key] = []

        # Populate the lists for each column
        for item in all_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question from the dataset
        question = self.data[idx][
            self.qk
        ]

        # Tokenize the question only
        input_ids, attention_mask = convert_raw_questions_to_model_questions(
            self.tokenizer, self.max_length, question, self.configs
        )

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask
        }
#================QAForgetEdgeDataset===============
class QAForgetEdgeDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        configs,
        max_length=512,
        question_key="question", 
        answer_key="answer",
    ):
        super(QAForgetEdgeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs
        all_QA = dataset_format_converstion(data_path, key="forget")

        QA_dict = {"question": [], "answer": [], "topic": []}
        for item in all_QA:
            QA_dict["question"].append(item["question"]) 
            QA_dict["answer"].append(item["answer"])
            QA_dict["topic"].append(item["topic"]) # @@@@
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "label": torch.stack(label_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
            "topic": self.data[idx].get("topic", ""), # @@@@
            "gt": self.data[idx].get("answer", ""), # @@@@
        }
#================QARetainedEdgeDataset===============
class QARetainedEdgeDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        configs,
        max_length=512,
        question_key="question", 
        answer_key="answer",
    ):
        super(QARetainedEdgeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs
        all_QA = dataset_format_converstion(data_path, key="retain")

        QA_dict = {"question": [], "answer": [], "topic": []}
        for item in all_QA:
            QA_dict["question"].append(item["question"]) 
            QA_dict["answer"].append(item["answer"])
            QA_dict["topic"].append(item["topic"])
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "label": torch.stack(label_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
            "topic": self.data[idx].get("topic", ""), # @@@@
            "gt": self.data[idx].get("answer", ""), # @@@@
        }
#================QAFactualDataset===============
class QAFactualDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        configs,
        max_length=512,
        question_key="question",
        answer_key="answer",
    ):
        super(QAFactualDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs
        all_QA = dataset_format_converstion(data_path, key="factual")

        QA_dict = {"question": [], "answer": [], "topic": []} # @ 수정
        for item in all_QA:
            QA_dict["question"].append(item["question"]) 
            QA_dict["answer"].append(item["answer"])
            QA_dict["topic"].append(item["topic"]) # @@@@
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "label": torch.stack(label_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
            "topic": self.data[idx].get("topic", ""), # @@@@
            "gt": self.data[idx].get("answer", ""), # @@@@
        }


#================TofuDatasets===============
class TofuForgetEdgeDataset(QAForgetEdgeDataset):
    def __init__(
        self, 
        data_path, 
        tokenizer, 
        configs, 
        max_length=512
    ):
        super().__init__(data_path, tokenizer, configs, max_length)
        all_QA = dataset_format_converstion_tofu(data_path, key="forget")
        QA_dict = {
            "question": [], "answer": [], "topic": [],
            "paraphrased_answer": [], "perturbed_answer": [], 
            "paraphrased_question": [], "author_idx": [], "forget_author": []
        }
        for item in all_QA:
            QA_dict["question"].append(item["question"]) 
            QA_dict["answer"].append(item["answer"])
            QA_dict["topic"].append(item["topic"])
            QA_dict["paraphrased_answer"].append(item["paraphrased_answer"])
            QA_dict["perturbed_answer"].append(item["perturbed_answer"])
            QA_dict["paraphrased_question"].append(item["paraphrased_question"])
            QA_dict["author_idx"].append(item["author_idx"])
            QA_dict["forget_author"].append(item["forget_author"])
        self.data = HFDataset.from_dict(QA_dict)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res.update({
            "paraphrased_answer": self.data[idx].get("paraphrased_answer", ""),
            "perturbed_answer": self.data[idx].get("perturbed_answer", []),
            "paraphrased_question": self.data[idx].get("paraphrased_question", ""),
            "author_idx": self.data[idx].get("author_idx"),
            "forget_author": self.data[idx].get("forget_author"),
        })
        return res

class TofuRetainedEdgeDataset(QARetainedEdgeDataset):
    def __init__(self, data_path, tokenizer, configs, max_length=512):
        super().__init__(data_path, tokenizer, configs, max_length)
        all_QA = dataset_format_converstion_tofu(data_path, key="retain")
        QA_dict = {
            "question": [], "answer": [], "topic": [],
            "paraphrased_answer": [], "perturbed_answer": [], 
            "paraphrased_question": [], "author_idx": [], "forget_author": []
        }
        for item in all_QA:
            QA_dict["question"].append(item["question"]) 
            QA_dict["answer"].append(item["answer"])
            QA_dict["topic"].append(item["topic"])
            QA_dict["paraphrased_answer"].append(item["paraphrased_answer"])
            QA_dict["perturbed_answer"].append(item["perturbed_answer"])
            QA_dict["paraphrased_question"].append(item["paraphrased_question"])
            QA_dict["author_idx"].append(item["author_idx"])
            QA_dict["forget_author"].append(item["forget_author"])
        self.data = HFDataset.from_dict(QA_dict)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res.update({
            "paraphrased_answer": self.data[idx].get("paraphrased_answer", ""),
            "perturbed_answer": self.data[idx].get("perturbed_answer", []),
            "paraphrased_question": self.data[idx].get("paraphrased_question", ""),
            "author_idx": self.data[idx].get("author_idx"),
            "forget_author": self.data[idx].get("forget_author"),
        })
        return res

class TofuFactualDataset(QAFactualDataset):
    def __init__(self, data_path, tokenizer, configs, max_length=512):
        super().__init__(data_path, tokenizer, configs, max_length)
        all_QA = dataset_format_converstion_tofu(data_path, key="factual")
        QA_dict = {
            "question": [], "answer": [], "topic": [],
            "paraphrased_answer": [], "perturbed_answer": [], 
            "paraphrased_question": [], "author_idx": [], "forget_author": []
        }
        for item in all_QA:
            QA_dict["question"].append(item["question"]) 
            QA_dict["answer"].append(item["answer"])
            QA_dict["topic"].append(item["topic"])
            QA_dict["paraphrased_answer"].append(item["paraphrased_answer"])
            QA_dict["perturbed_answer"].append(item["perturbed_answer"])
            QA_dict["paraphrased_question"].append(item["paraphrased_question"])
            QA_dict["author_idx"].append(item["author_idx"])
            QA_dict["forget_author"].append(item["forget_author"])
        self.data = HFDataset.from_dict(QA_dict)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res.update({
            "paraphrased_answer": self.data[idx].get("paraphrased_answer", ""),
            "perturbed_answer": self.data[idx].get("perturbed_answer", []),
            "paraphrased_question": self.data[idx].get("paraphrased_question", ""),
            "author_idx": self.data[idx].get("author_idx"),
            "forget_author": self.data[idx].get("forget_author"),
        })
        return res

############################################################################
# def custom_question_collator(samples):
#     input_ids = [s["input_ids"] for s in samples]
#     attention_mask = [s["attention_mask"] for s in samples]
#     return torch.stack(input_ids), torch.stack(attention_mask)


# def custom_data_collator_forget(samples):
#     forget_samples, retain_samples = (
#         [sample[0] for sample in samples],
#         [sample[1] for sample in samples],
#     )
#     rets = []
#     for data_type in ["forget", "retain"]:
#         data = forget_samples if data_type == "forget" else retain_samples
#         input_ids = [s[0] for s in data]
#         labels = [s[1] for s in data]
#         attention_mask = [s[2] for s in data]
#         rets.append(
#             (torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask))
#         )
#     return rets
############################################################################
def custom_tofu_collator(samples):
    input_ids = [s["input_ids"] for s in samples]
    labels = [s["label"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    topics = [s.get("topic", "") for s in samples]
    gts = [s.get("gt", "") for s in samples]
    # 5가지 data stack 
    paraphrased_answers = [s.get("paraphrased_answer", "") for s in samples]
    perturbed_answers = [s.get("perturbed_answer", []) for s in samples]
    paraphrased_questions = [s.get("paraphrased_question", "") for s in samples]
    author_idxs = [s.get("author_idx") for s in samples]
    forget_authors = [s.get("forget_author") for s in samples]
    return (
        torch.stack(input_ids),
        torch.stack(labels),
        torch.stack(attention_mask),
        topics,
        gts,
        paraphrased_answers,
        perturbed_answers,
        paraphrased_questions,
        author_idxs,
        forget_authors
    )

def custom_qa_collator(samples):
    input_ids = [s["input_ids"] for s in samples]
    labels = [s["label"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    topics = [s.get("topic", "") for s in samples]
    gts = [s.get("gt", "") for s in samples] # @@@@
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), topics, gts

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous() # [16, 511]
    output = output[..., :-1, :].contiguous() # [16, 511, 128256]

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none") # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss
