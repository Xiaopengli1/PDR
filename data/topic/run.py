from datasets import load_dataset

dataset = load_dataset("LongLaMP/LongLaMP", name = "topic_writing_user")

# "product_review_user"
# "topic_writing_user"

splits = ["train", "val", "test"]

for split in splits:
    data_split = dataset[split]

    filtered_data = data_split.filter(lambda example: len(example["output"])>1000 )

    output_filename = f"{split}_output_gt1500.json"

    filtered_data.to_json(output_filename)

    print(f"{split} is already success saved.")