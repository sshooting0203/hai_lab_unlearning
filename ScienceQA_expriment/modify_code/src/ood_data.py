import datasets, random, json, os, torch 
from datasets import load_dataset
from torch.utils.data import DataLoader

datasets.logging.set_verbosity(datasets.logging.ERROR)

def load(task_name, tokenizer, max_seq_length=512, is_id=False, base_unlearn_path=None, base_ood_path=None): # 256
    sentence1_key, sentence2_key = ("text_input", None)
    if "scienceqa" in task_name and "ood" not in task_name:
        datasets = load_scienceqa_topic(task_name.split("_")[-1],  base_unlearn_path)
    elif "ood_scienceqa" in task_name:
        datasets = load_scienceqa_notopic(base_ood_path)
    elif "clinc_qa" in task_name and "ood" not in task_name:
        datasets = load_clinc_qa_topic(task_name.split("_")[-1],  base_unlearn_path)
    elif "ood_clinc_qa" in task_name:
        datasets = load_clinc_qa_notopic(base_ood_path)
    elif "wmdp" in task_name and "ood" not in task_name:
        datasets = load_wmdp_topic(task_name.split("_")[-1], base_unlearn_path)
    elif "ood_wmdp" in task_name:
        datasets = load_wmdp_notopic(base_ood_path)
    elif "scicombine" in task_name and "ood" not in task_name:
        datasets = load_scicombine_topic(task_name.split("_")[-1], base_unlearn_path)
    elif "ood_scicombine" in task_name:
        datasets = load_scicombine_notopic(base_ood_path)

    def preprocess_function(examples):
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, padding='max_length', max_length=max_seq_length, truncation=True)
        return result

    train_dataset = list(map(preprocess_function, datasets['train'])) if 'train' in datasets and is_id else None
    test_dataset = list(map(preprocess_function, datasets['test'])) if 'test' in datasets else None
    return train_dataset, test_dataset

def load_scicombine_topic(topic, path):
    with open(path + f"scienceqa_{topic}_train.json") as fp:
        train_data = json.load(fp)
    forget_ratio = 0.8
    length = forget_ratio * len(train_data)
    forget_index_list = random.sample(range(len(train_data)), int(length))
    retain_index_list = list(set(range(len(train_data))) - set(forget_index_list))
    train_dataset = [train_data[i] for i in forget_index_list]
    test_dataset = [train_data[i] for i in retain_index_list]
    datasets = {'train': train_dataset,  'test': test_dataset}
    return datasets

def load_scicombine_notopic(path):
    with open(path + f"_train_RD.json") as fp:
        train_dataset = json.load(fp)
    with open(path + f"_validation_RD.json") as fp:
        dev_dataset = json.load(fp)
    datasets = {'train': train_dataset, 'test': train_dataset}
    return datasets

def load_wmdp_topic(topic, path):
    with open(path + f"wmdp_{topic}_train.json") as fp:
        train_data = json.load(fp)
    forget_ratio = 0.8
    length = forget_ratio * len(train_data)
    forget_index_list = random.sample(range(len(train_data)), int(length))
    retain_index_list = list(set(range(len(train_data))) - set(forget_index_list))
    train_dataset = [train_data[i] for i in forget_index_list]
    test_dataset = [train_data[i] for i in retain_index_list]
    datasets = {'train': train_dataset,  'test': test_dataset}
    return datasets

def load_wmdp_notopic(path):
    with open(path + f"_train_RD.json") as fp:
        train_dataset = json.load(fp)
    with open(path + f"_validation_RD.json") as fp:
        dev_dataset = json.load(fp)
    datasets = {'train': train_dataset, 'test': train_dataset}
    return datasets

### CLINC150 
def load_clinc_qa_topic(topic, path):
    fp = os.path.join(path, f"clincqa_{topic}_train.json")
    with open(fp, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    forget_ratio = 0.8
    length = forget_ratio * len(train_data)
    forget_index_list = random.sample(range(len(train_data)), int(length))
    retain_index_list = list(set(range(len(train_data))) - set(forget_index_list))
    train_dataset = [train_data[i] for i in forget_index_list]
    test_dataset = [train_data[i] for i in retain_index_list]
    datasets = {'train': train_dataset,  'test': test_dataset}
    return datasets

def load_clinc_qa_notopic(path):
    with open(path + f"_train_RD.json") as fp:
        train_dataset = json.load(fp)
    with open(path + f"_validation_RD.json") as fp:
        dev_dataset = json.load(fp)
    datasets = {'train': train_dataset, 'test': train_dataset}
    return datasets

### ScienceQA
def load_scienceqa_topic(topic, path):
    print("load_scienceqa_topic", path, topic )
    with open(path + f"_train_SD.json") as fp:
        train_data = json.load(fp)

    forget_ratio = 0.8
    length = forget_ratio * len(train_data)
    forget_index_list = random.sample(range(len(train_data)), int(length))
    retain_index_list = list(set(range(len(train_data))) - set(forget_index_list))

    train_dataset = [train_data[i] for i in forget_index_list]
    test_dataset = [train_data[i] for i in retain_index_list]
    datasets = {'train': train_dataset,  'test': test_dataset}
    return datasets

def load_scienceqa_notopic(path):
    with open(path + f"_train_RD.json") as fp:
        train_dataset = json.load(fp)
    with open(path + f"_validation_RD.json") as fp:
        dev_dataset = json.load(fp)
    datasets = {'train': train_dataset, 'test': train_dataset}
    return datasets
