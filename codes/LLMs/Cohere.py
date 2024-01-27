import cohere
co = cohere.Client()

def cohereModel(dataset):
  '''
    dataset: list of Dicts with keys: 'sentence', 'paragraph' (has the sentence replaces with [MASK])
  '''
  outputs = []
  for i in range(len(dataset)):
    prompt = f"""
    Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
    """
    response = co.generate(
      prompt
    )
    outputs.append(response)
  return outputs
