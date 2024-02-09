from datasets import load_from_disk, load_dataset

test = load_from_disk('../datasets/SubtaskB/test2')

test2 = load_dataset('json', data_files='./subtaskB(2).jsonl', split='train')

print(len(test))
print(len(test2))


labels = test2['label']
test = test.add_column('label', labels)

test.save_to_disk('../datasets/SubtaskB/test3')