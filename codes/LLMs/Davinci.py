import openai
# Set OpenAI API key
api_key = "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"
openai.api_key = api_key

def get_completion(prompts, model="davinci-002"):
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0, # The degree of randomness of the model's output
    )
    return response.choices[0].text.strip()
import pickle
def davinciModel(dataset):
    '''
        dataset: list of Dicts with keys: 'sentence', 'paragraph' (has the sentence replaces with [MASK])
    '''
    outputs = []
    # with open('davinciOutputs.pkl', 'rb') as f:
    #     outputs = pickle.load(f)

    

    for i in range(len(dataset)):
        if i < len(outputs):
            continue
        prompt = f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
        """
        response = get_completion(prompt)
        outputs.append(response)
        print(f"davinci is at {i} of {len(dataset)}", end='\r')
        if i % 100 == 0:
            pickle.dump(outputs, open("davinciOutputs.pkl", "wb"))
    return outputs
