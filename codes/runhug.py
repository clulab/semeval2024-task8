import random
random.seed(68)
from datasets import load_dataset, Dataset

from LLMs.Bloomz import BloomzModel
# from LLMs.ChatGPT import gptModel
# from LLMs.Cohere import cohereModel
from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel


train = load_dataset('json', data_files='../datasets/SubtaskB/trainMasks.json', split='train')
dev = load_dataset('json', data_files='../datasets/SubtaskB/devMasks.json', split='train')
test = load_dataset('json', data_files='../datasets/SubtaskB/testMasks.json', split='train')



# run bloomz
trainOutpus = BloomzModel(train)
train.add_column('bloomz', trainOutpus)
train.save_to_disk('./datasets/SubtaskB/trainBloomz')


devOutputs = BloomzModel(dev)
dev.add_column('bloomz', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devBloomz')

testOutputs = BloomzModel(test)

test.add_column('bloomz', testOutputs)
test.save_to_disk('./datasets/SubtaskB/testBloomz')
# add a column to the dataset with the outputs

# save the datasets

# load the datasets
train = load_dataset('json', data_files='../datasets/SubtaskB/trainMasks.json', split='train')
trainOutpus = DollyModel(train)
train.add_column('dolly', trainOutpus)
train.save_to_disk('./datasets/SubtaskB/trainDolly')


dev = load_dataset('json', data_files='../datasets/SubtaskB/devMasks.json', split='train')

# run dolly
devOutputs = DollyModel(dev)
dev.add_column('dolly', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devDolly')

test = load_dataset('json', data_files='../datasets/SubtaskB/testMasks.json', split='train')
testOutputs = DollyModel(test)

# add a column to the dataset with the outputs
test.add_column('dolly', testOutputs)

# save the datasets
test.save_to_disk('./datasets/SubtaskB/testDolly')
