from datasets import load_from_disk

train = load_from_disk('./datasets/SubtaskB/trainDavinci')
dev = load_from_disk('./datasets/SubtaskB/devDavinci')
test = load_from_disk('./datasets/SubtaskB/testDavinci')

print(train)

# remove the 'gpt' column
# train = train.remove_columns('davinci')
# dev = dev.remove_columns('davinci')
# test = test.remove_columns('davinci')

# save the datasets
train.save_to_disk('../datasets/SubtaskB/train')
dev.save_to_disk('../datasets/SubtaskB/dev')
test.save_to_disk('../datasets/SubtaskB/test')