import random
random.seed(68)
from datasets import load_from_disk

# from LLMs.Bloomz import BloomzModel
from LLMs.ChatGPT22 import gptModel
# from LLMs.Cohere import cohereModel
# from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel
'''
train = load_from_disk('../datasets/SubtaskB/train')
trainOutputs = gptModel(train)
train = train.add_column('gpt', trainOutputs)
train.save_to_disk('./datasets/SubtaskB/trainGPT')
'''
dev = load_from_disk('../datasets/SubtaskB/dev')
devOutputs = gptModel(dev)
dev = dev.add_column('gpt', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devGPT')
