import openai
# Set OpenAI API key
api_key = "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"
openai.api_key = api_key
import pickle

def get_completion(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": message}]
    )
    return response.choices[0].message['content'].strip()

def gptModel(dataset):
    '''
        dataset: list of Dicts with keys: 'sentence', 'paragraph' (has the sentence replaces with [MASK])
    '''
    outputs = []
    with open('gptOutputs.pkl', 'rb') as f:
        outputs = pickle.load(f)
    
    for i in range(len(dataset)):
        if i < len(outputs):
            continue
        # prompt = f"""
        # Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
        # """
        prompt = f"""
        {dataset[i]['sentence']} \n rewrite the above sentence in a different way.
        """
        try:
            response = get_completion(prompt)
        except Exception as e:
            print(e)
            response = ''


        outputs.append(response)
        print(f"gpt is at {i} of {len(dataset)}", end='\r')
        if i % 100 == 0:
            pickle.dump(outputs, open("gptOutputs.pkl", "wb"))
    return outputs
