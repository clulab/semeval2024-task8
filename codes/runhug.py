import random
random.seed(68)
from datasets import load_from_disk

from LLMs.Bloomz import BloomzModel
# from LLMs.ChatGPT import gptModel
# from LLMs.Cohere import cohereModel
from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel


train = load_from_disk('../datasets/SubtaskB/train')
dev = load_from_disk('../datasets/SubtaskB/dev')
test = load_from_disk('../datasets/SubtaskB/test')


# run bloomz
trainOutpus = BloomzModel(train)
train = train.add_column('bloomz', trainOutpus)
train.save_to_disk('./datasets/SubtaskB/trainBloomz')


devOutputs = BloomzModel(dev)
dev = dev.add_column('bloomz', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devBloomz')

testOutputs = BloomzModel(test)

test = test.add_column('bloomz', testOutputs)
test.save_to_disk('./datasets/SubtaskB/testBloomz')
# add a column to the dataset with the outputs

# save the datasets

# load the datasets
train = load_from_disk('../datasets/SubtaskB/train')
trainOutpus = DollyModel(train)
train = train.add_column('dolly', trainOutpus)
train.save_to_disk('./datasets/SubtaskB/trainDolly')


dev = load_from_disk('../datasets/SubtaskB/dev')

# run dolly
devOutputs = DollyModel(dev)
dev.add_column('dolly', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devDolly')

test = load_from_disk('../datasets/SubtaskB/test')
testOutputs = DollyModel(test)

# add a column to the dataset with the outputs
test = test.add_column('dolly', testOutputs)

# save the datasets
test.save_to_disk('./datasets/SubtaskB/testDolly')
