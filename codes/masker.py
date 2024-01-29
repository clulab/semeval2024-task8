import random
random.seed(68)


from datasets import load_dataset, Dataset, load_from_disk
import spacy
nlp = spacy.load("en_core_web_sm")

# train = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_train.jsonl', split='train')
train = load_from_disk('../datasets/SubtaskB/train')
dev = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_dev.jsonl', split='train')
test = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_test.jsonl', split='train')

trainMasks = []
devMasks = []
testMasks = []
def masker(dev, devMasks): 
    for i in range(len(dev)):
        paragraph = dev[i]['text']
        doc = nlp(paragraph)
        # maskeds = []

        lenSent = 0
        for sent in doc.sents:
            lenSent += 1
        
        if lenSent > 2:
            # first = 1
            midRand = random.randint(2, lenSent-1)
            # last = lenSent

            # j = 0
            found = False
            tryNum = 0
            while found is False:
                j = 0
                for sent in doc.sents:
                    if j == midRand:
                        if len(sent.text) > 12 or tryNum > 9:
                            sentDict = {}
                            sentDict['sentence'] = sent.text
                            sentDict['idx'] = str(dev[i]['id']) + '_' + str(j)
                            # replace the sentence in 'text' with [MASK]
                            paragraph = paragraph.replace(sent.text, '[MASK]')
                            sentDict['paragraph'] = paragraph
                            devMasks.append(sentDict)
                            found = True
                        else:
                            midRand = random.randint(2, lenSent-1)
                    if found:
                        break
                    j += 1
                tryNum += 1
            assert found is True, f"{tryNum} tries"

        else:
            # choose the last sentence no matter what
            j = 0
            for sent in doc.sents:
                if j == lenSent-1:
                    sentDict = {}
                    sentDict['sentence'] = sent.text
                    sentDict['idx'] = str(dev[i]['id']) + '_' + str(j)
                    # replace the sentence in 'text' with [MASK]
                    paragraph = paragraph.replace(sent.text, '[MASK]')
                    sentDict['paragraph'] = paragraph
                    devMasks.append(sentDict)
                j += 1
        print(f"finished extracting {i}th paragraph from dev set out of {len(dev)} paragraphs", end='\r')
        assert len(devMasks) == i+1
    return devMasks

import json
devMasks = masker(dev, devMasks)
assert len(devMasks) == len(dev)
with open('../datasets/SubtaskB/devMasks.json', 'w') as f:
    for item in devMasks:
        f.write(json.dumps(item) + '\n')

testMasks = masker(test, testMasks)
assert len(testMasks) == len(test)
with open('../datasets/SubtaskB/testMasks.json', 'w') as f:
    for item in testMasks:
        f.write(json.dumps(item) + '\n')

trainMasks = masker(train, trainMasks)
assert len(trainMasks) == len(train)
with open('../datasets/SubtaskB/trainMasks.json', 'w') as f:
    for item in trainMasks:
        f.write(json.dumps(item) + '\n')