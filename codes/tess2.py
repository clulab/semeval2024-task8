import pickle

with open('cohereOutputs2.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data))