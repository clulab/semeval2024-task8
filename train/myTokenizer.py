# 71026
import random
random.seed(42)

def randomNum(max, num):
    '''
    This function generates a list of num rdifferent random numbers from 0 to max-1.
    '''
    result = []
    while len(result) < num:
        temp = random.randint(0, max-1)
        if temp not in result:
            result.append(temp)
    return result

import pickle
for i in [j for j in range(1, 7)]:
    with open('./trainIDs' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(randomNum(71026, i * 10000), f)
    print(f"finished {i}th file", end='\r')


