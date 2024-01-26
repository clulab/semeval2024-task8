import random
random.seed(68)


from datasets import load_dataset, Dataset
import spacy
nlp = spacy.load("en_core_web_sm")

train = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_train.jsonl', split='train')
dev = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_dev.jsonl', split='train')
test = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_test.jsonl', split='train')

trainMasks = []
devMasks = []
testMasks = []

for i in range(len(dev)):
    paragraph = dev[i]['text']
    doc = nlp(paragraph)
    # maskeds = []

    lenSent = 1
    for sent in doc.sents:
        lenSent += 1
    
    if lenSent > 2:
        first = 1
        midRand = random.randint(2, lenSent-1)
        last = lenSent

        j = 0
        for sent in doc.sents:
            if j == first or j == midRand or j == last:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(dev[i]['id']) + '_' + str(j)
                devMasks.append(sentDict)
            j += 1
    elif lenSent == 2:
        first = 1
        last = lenSent

        j = 0
        for sent in doc.sents:
            if j == first or j == last:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(dev[i]['id']) + '_' + str(j)
                devMasks.append(sentDict)
            j += 1
    else:
        first = 1

        j = 0
        for sent in doc.sents:
            if j == first:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(dev[i]['id']) + '_' + str(j)
                devMasks.append(sentDict)
            j += 1
    print(f"finished extracting {i}th paragraph from dev set out of {len(dev)} paragraphs", end='\r')
import json
# devMasks = Dataset.from_dict(devMasks)
# devMasks.save_to_disk('../datasets/SubtaskB/devMasks')
with open('../datasets/SubtaskB/devMasks.json', 'w') as f:
    for item in devMasks:
        f.write(json.dumps(item) + '\n')




for i in range(len(train)):
    paragraph = train[i]['text']
    doc = nlp(paragraph)
    # maskeds = []

    lenSent = 1
    for sent in doc.sents:
        lenSent += 1
    
    if lenSent > 2:
        first = 1
        midRand = random.randint(2, lenSent-1)
        last = lenSent

        j = 0
        for sent in doc.sents:
            if j == first or j == midRand or j == last:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(train[i]['id']) + '_' + str(j)
                trainMasks.append(sentDict)
            j += 1
    elif lenSent == 2:
        first = 1
        last = lenSent

        j = 0
        for sent in doc.sents:
            if j == first or j == last:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(train[i]['id']) + '_' + str(j)
                trainMasks.append(sentDict)
            j += 1
    else:
        first = 1

        j = 0
        for sent in doc.sents:
            if j == first:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(train[i]['id']) + '_' + str(j)
                trainMasks.append(sentDict)
            j += 1
    print(f"finished extracting {i}th paragraph from train set out of {len(train)} paragraphs", end='\r')
    
# trainMasks = Dataset.from_dict(trainMasks)
import json
with open('../datasets/SubtaskB/trainMasks.json', 'w') as f:
    for item in trainMasks:
        f.write(json.dumps(item) + '\n')
# trainMasks.save_to_disk('../datasets/SubtaskB/trainMasks')
    



for i in range(len(test)):
    paragraph = test[i]['text']
    doc = nlp(paragraph)
    # maskeds = []

    lenSent = 1
    for sent in doc.sents:
        lenSent += 1
    
    if lenSent > 2:
        first = 1
        midRand = random.randint(2, lenSent-1)
        last = lenSent

        j = 0
        for sent in doc.sents:
            if j == first or j == midRand or j == last:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(test[i]['id']) + '_' + str(j)
                testMasks.append(sentDict)
            j += 1
    elif lenSent == 2:
        first = 1
        last = lenSent

        j = 0
        for sent in doc.sents:
            if j == first or j == last:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(test[i]['id']) + '_' + str(j)
                testMasks.append(sentDict)
            j += 1
    else:
        first = 1

        j = 0
        for sent in doc.sents:
            if j == first:
                sentDict = {}
                sentDict['sentence'] = sent.text
                sentDict['idx'] = str(test[i]['id']) + '_' + str(j)
                testMasks.append(sentDict)
            j += 1
    print(f"finished extracting {i}th paragraph from test set out of {len(test)} paragraphs", end='\r')

with open('../datasets/SubtaskB/testMasks.json', 'w') as f:
    for item in testMasks:
        f.write(json.dumps(item) + '\n')
