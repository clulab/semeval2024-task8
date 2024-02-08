import random
random.seed(68)
from datasets import load_from_disk

# from LLMs.Bloomz import BloomzModel
from LLMs.ChatGPT2 import gptModel
# from LLMs.Cohere import cohereModel
# from LLMs.Dolly import DollyModel
# from LLMs.Davinci import davinciModel

test = load_from_disk('../datasets/SubtaskB/test')
testOutputs = gptModel(test)
test = test.add_column('gpt', testOutputs)
test.save_to_disk('./datasets/SubtaskB/testGPT')