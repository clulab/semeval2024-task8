import random
random.seed(68)
from datasets import load_dataset, Dataset

# from LLMs.Bloomz import BloomzModel
# from LLMs.ChatGPT import gptModel
from LLMs.Cohere2 import cohereModel
# from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel


# train = load_dataset('json', data_files='../datasets/SubtaskB/trainMasks.json', split='train')
# dev = load_dataset('json', data_files='../datasets/SubtaskB/devMasks.json', split='train')
# devOutputs = cohereModel(dev)
# dev.add_column('cohere', devOutputs)
test = load_dataset('json', data_files='../datasets/SubtaskB/testMasks.json', split='train')
testOutputs = cohereModel(test)
test.add_column('cohere', testOutputs)
test.save_to_disk('../datasets/SubtaskB/testCohere')

'''

# dataset has features ['idx', 'paragraph', 'sentence']
print('cohere done')
# add a column to the dataset with the outputs

# save the datasets
dev.save_to_disk('../datasets/SubtaskB/devCohere')

# load the datasets
train = load_dataset('./datasets/SubtaskB/trainCohere', split='train')
dev = load_dataset('./datasets/SubtaskB/devCohere', split='train')
test = load_dataset('./datasets/SubtaskB/testCohere', split='train')

# run davinci
trainOutpus = davinciModel(train)
devOutputs = davinciModel(dev)
testOutputs = davinciModel(test)
print('davinci done')
# add a column to the dataset with the outputs
train.add_column('davinci', trainOutpus)
dev.add_column('davinci', devOutputs)
test.add_column('davinci', testOutputs)

# save the datasets
train.save_to_disk('./datasets/SubtaskB/trainDavinci')
dev.save_to_disk('./datasets/SubtaskB/devDavinci')
test.save_to_disk('./datasets/SubtaskB/testDavinci')

# load the datasets
train = load_dataset('./datasets/SubtaskB/trainDavinci', split='train')
dev = load_dataset('./datasets/SubtaskB/devDavinci', split='train')
test = load_dataset('./datasets/SubtaskB/testDavinci', split='train')

# run chatgpt
trainOutpus = gptModel(train)
devOutputs = gptModel(dev)
testOutputs = gptModel(test)
print('gpt done')
# add a column to the dataset with the outputs
train.add_column('gpt', trainOutpus)
dev.add_column('gpt', devOutputs)
test.add_column('gpt', testOutputs)

# save the datasets
train.save_to_disk('./datasets/SubtaskB/trainGPT')
dev.save_to_disk('./datasets/SubtaskB/devGPT')
test.save_to_disk('./datasets/SubtaskB/testGPT')

# load the datasets
train = load_dataset('./datasets/SubtaskB/trainGPT', split='train')
dev = load_dataset('./datasets/SubtaskB/devGPT', split='train')
test = load_dataset('./datasets/SubtaskB/testGPT', split='train')

# run bloomz
trainOutpus = BloomzModel(train)
devOutputs = BloomzModel(dev)
testOutputs = BloomzModel(test)

# add a column to the dataset with the outputs
train.add_column('bloomz', trainOutpus)
dev.add_column('bloomz', devOutputs)
test.add_column('bloomz', testOutputs)

# save the datasets
train.save_to_disk('./datasets/SubtaskB/trainBloomz')
dev.save_to_disk('./datasets/SubtaskB/devBloomz')
test.save_to_disk('./datasets/SubtaskB/testBloomz')

# load the datasets
train = load_dataset('./datasets/SubtaskB/trainBloomz', split='train')
dev = load_dataset('./datasets/SubtaskB/devBloomz', split='train')
test = load_dataset('./datasets/SubtaskB/testBloomz', split='train')

# run dolly
trainOutpus = DollyModel(train)
devOutputs = DollyModel(dev)
testOutputs = DollyModel(test)

# add a column to the dataset with the outputs
train.add_column('dolly', trainOutpus)
dev.add_column('dolly', devOutputs)
test.add_column('dolly', testOutputs)

# save the datasets
train.save_to_disk('./datasets/SubtaskB/trainDolly')
dev.save_to_disk('./datasets/SubtaskB/devDolly')
test.save_to_disk('./datasets/SubtaskB/testDolly')
'''