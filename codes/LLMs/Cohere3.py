import cohere
co = cohere.Client('vC3TMSOBRsbjdFDfstSyF6Ytct9AEHqVXHwYJCFz')
import pickle
def cohereModel(dataset):
  '''
    dataset: list of Dicts with keys: 'sentence', 'paragraph' (has the sentence replaces with [MASK])
  '''
  outputs = []
  with open('cohereOutputs3.pkl', 'rb') as f:
    outputs = pickle.load(f)

  for i in range(len(dataset)):
    if i < len(outputs):
      continue
    prompt = f"""
    Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
    """
    try:
      response = co.generate(
        prompt,
        truncate='END',
      )
    except Exception as e:
      print(e)
      response = ''
      

    outputs.append(response)
    print(f"cohere is at {i} of {len(dataset)}", end='\r')
    if i % 100 == 0:
      pickle.dump(outputs, open("cohereOutputs3.pkl", "wb"))
      
  return outputs