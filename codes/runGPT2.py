import random
random.seed(68)
from datasets import load_dataset, Dataset

# from LLMs.Bloomz import BloomzModel
from LLMs.ChatGPT22 import gptModel
# from LLMs.Cohere import cohereModel
# from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel

dev = load_dataset('json', data_files='../datasets/SubtaskB/devMasks.json', split='train')
devOutputs = gptModel(dev)
dev.add_column('gpt', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devGPT')
test = load_dataset('json', data_files='../datasets/SubtaskB/testMasks.json', split='train')

# run chatgpt
testOutputs = gptModel(test)
print('gpt done')
# add a column to the dataset with the outputs
test.add_column('gpt', testOutputs)

# save the datasets
test.save_to_disk('./datasets/SubtaskB/testGPT')
