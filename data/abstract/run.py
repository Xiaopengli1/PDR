# from datasets import load_dataset

# dataset = load_dataset("LongLaMP/LongLaMP", name = "abstract_generation_user")

# # "product_review_user"
# # "topic_writing_user"

# splits = ["train", "val", "test"]

# for split in splits:
#     data_split = dataset[split]

#     filtered_data = data_split.filter(lambda example: len(example["output"])>1000 )

#     output_filename = f"{split}_output_gt1500.json"

#     filtered_data.to_json(output_filename)

#     print(f"{split} is already success saved.")

from datasets import load_dataset
import os
import json


# Load abstract_generation_user subset
dataset = load_dataset("LongLaMP/LongLaMP", name="abstract_generation_user")



long_output_samples = []
for sample in tqdm(dataset["test"]):
    if len(sample['output']) > 5000:
        long_output_samples.append(sample)
    if len(long_output_samples) >= 20:
        break

for sample in tqdm(long_output_samples):
    reviewer_id = sample.get('reviewerId', 'unknown_reviewer') # Use .get() for safety
    input_content = sample.get('input', '')
    output_content = sample.get('output', '')
    profile_content = sample.get('profile', '')

    # Create user directory if it doesn't exist
    user_dir = reviewer_id
    os.makedirs(user_dir, exist_ok=True)

    # Create knowledge_base directory inside the user directory
    knowledge_base_dir = os.path.join(user_dir, 'knowledge_base')
    os.makedirs(knowledge_base_dir, exist_ok=True)


    # Convert the profile list to a JSON string before writing
    try:
        profile_string = json.dumps(profile_content, indent=4, ensure_ascii=False)
        # Write profile.txt to the knowledge_base directory
        with open(os.path.join(knowledge_base_dir, 'profile.txt'), 'w', encoding='utf-8') as f:
            f.write(profile_string)
    except TypeError as e:
        print(f"Could not serialize profile for reviewerId {reviewer_id}: {e}")


    # Write input and output to the user directory (or you could put these in knowledge_base too if needed)
    with open(os.path.join(user_dir, 'input.txt'), 'w', encoding='utf-8') as f:
        f.write(input_content)

    with open(os.path.join(user_dir, 'output.txt'), 'w', encoding='utf-8') as f:
        f.write(output_content)


print("Files generated successfully.")
