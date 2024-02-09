from datasets import load_dataset
import random

# Load the dataset
test = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_test.jsonl', split='train')

# Set the seed for reproducibility
random.seed(6985)

# Generate random predictions and add them as a new column
preds = [random.randint(0, 5) for _ in range(len(test))]
test = test.add_column('label', preds)
# test = test.remove_columns('sentence', 'paragraph')


# Save the modified dataset as a JSON file
output_file_path = './datasets/SubtaskB/testRandom.json'
test.to_json(output_file_path, orient='records')

# Output the path for downloading or further use
output_file_path
