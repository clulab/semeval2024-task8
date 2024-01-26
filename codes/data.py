from datasets import load_dataset

# import the jsonl file
train = load_dataset('json', data_files='datasets/SubtaskB/subtaskB_train.jsonl')['train']
test = load_dataset('json', data_files='datasets/SubtaskB/subtaskB_dev.jsonl')['train']

print(train)
print(test)