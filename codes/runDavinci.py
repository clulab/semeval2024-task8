import random
random.seed(68)
from datasets import Dataset, load_from_disk

# from LLMs.Bloomz import BloomzModel
# from LLMs.ChatGPT import gptModel
# from LLMs.Cohere import cohereModel
# from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel
from LLMs.Davinci import davinciModel


# train = load_dataset('json', data_files='../datasets/SubtaskB/trainMasks.json', split='train')
# train = load_dataset('json', data_files='../datasets/SubtaskB/trainMasks.json', split='train')
train = load_from_disk('../datasets/SubtaskB/train')

# run davinci
# trainOutpus = davinciModel(train)
# print(f"len(trainOutpus): {len(trainOutpus)}")
# print(f"len(train): {len(train)}")
# train.add_column('davinci', trainOutpus)
# train.save_to_disk('./datasets/SubtaskB/trainDavinci')

# trainOutputs = davinciModel(train)
# train.add_column('davinci', trainOutputs)
# train.save_to_disk('./datasets/SubtaskB/trainDavinci')

testOutputs = davinciModel(train)
train = train.add_column('davinci', testOutputs)
train.save_to_disk('./datasets/SubtaskB/trainDavinci')
print('davinci done')
# add a column to the dataset with the outputs

# save the datasets



# print(len(outputs))
