import random
random.seed(68)
from datasets import load_dataset, Dataset

# from LLMs.Bloomz import BloomzModel
# from LLMs.ChatGPT import gptModel
# from LLMs.Cohere import cohereModel
# from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel
from LLMs.Davinci import davinciModel


# train = load_dataset('json', data_files='../datasets/SubtaskB/trainMasks.json', split='train')
dev = load_dataset('json', data_files='../datasets/SubtaskB/devMasks.json', split='train')
test = load_dataset('json', data_files='../datasets/SubtaskB/testMasks.json', split='train')

# run davinci
# trainOutpus = davinciModel(train)
# print(f"len(trainOutpus): {len(trainOutpus)}")
# print(f"len(train): {len(train)}")
# train.add_column('davinci', trainOutpus)
# train.save_to_disk('./datasets/SubtaskB/trainDavinci')

devOutputs = davinciModel(dev)
dev.add_column('davinci', devOutputs)
dev.save_to_disk('./datasets/SubtaskB/devDavinci')

testOutputs = davinciModel(test)
test.add_column('davinci', testOutputs)
test.save_to_disk('./datasets/SubtaskB/testDavinci')
print('davinci done')
# add a column to the dataset with the outputs

# save the datasets



# print(len(outputs))
