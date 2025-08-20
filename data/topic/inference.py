from datasets import load_dataset

data_files = {
    "train": "train.json",
    "val": "val.json",
    "test": "test.json",
}

dataset = load_dataset("json", data_file = data_files )
