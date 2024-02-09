from datasets import load_from_disk

train = load_from_disk('../datasets/SubtaskB/train')
dev = load_from_disk('../datasets/SubtaskB/dev')
test = load_from_disk('../datasets/SubtaskB/test')

trainBloomz = load_from_disk('./datasets/SubtaskB/trainBloomz')
trainDavinci = load_from_disk('./datasets/SubtaskB/trainDavinci')
trainGPT = load_from_disk('./datasets/SubtaskB/trainGPT')
trainCohere = load_from_disk('./datasets/SubtaskB/trainCohere')

devBloomz = load_from_disk('./datasets/SubtaskB/devBloomz')
devDavinci = load_from_disk('./datasets/SubtaskB/devDavinci')
devGPT = load_from_disk('./datasets/SubtaskB/devGPT')
devCohere = load_from_disk('./datasets/SubtaskB/devCohere')

testBloomz = load_from_disk('./datasets/SubtaskB/testBloomz')
testDavinci = load_from_disk('./datasets/SubtaskB/testDavinci')
testGPT = load_from_disk('./datasets/SubtaskB/testGPT')
testCohere = load_from_disk('./datasets/SubtaskB/testCohere')

trainBloomzList = trainBloomz['bloomz']
trainDavinciList = trainDavinci['davinci']
trainGPTList = trainGPT['gpt']
trainCohereList = trainCohere['cohere']

devBloomzList = devBloomz['bloomz']
devDavinciList = devDavinci['davinci']
devGPTList = devGPT['gpt']
devCohereList = devCohere['cohere']

testBloomzList = testBloomz['bloomz']
testDavinciList = testDavinci['davinci']
testGPTList = testGPT['gpt']
testCohereList = testCohere['cohere']

train = train.add_column('bloomz', trainBloomzList)
train = train.add_column('davinci', trainDavinciList)
train = train.add_column('gpt', trainGPTList)
train = train.add_column('cohere', trainCohereList)

dev = dev.add_column('bloomz', devBloomzList)
dev = dev.add_column('davinci', devDavinciList)
dev = dev.add_column('gpt', devGPTList)
dev = dev.add_column('cohere', devCohereList)

test = test.add_column('bloomz', testBloomzList)
test = test.add_column('davinci', testDavinciList)
test = test.add_column('gpt', testGPTList)
test = test.add_column('cohere', testCohereList)

train.save_to_disk('../datasets/SubtaskB/train2')
dev.save_to_disk('../datasets/SubtaskB/dev2')
test.save_to_disk('../datasets/SubtaskB/test2')

