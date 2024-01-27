import openai
# Set OpenAI API key
api_key = "Sam Altman's Dad"
openai.api_key = api_key

def get_completion(prompts, model="davinci-002"):
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0, # The degree of randomness of the model's output
    )
    return response.choices[0].text.strip()

def davinciModel(dataset):
    '''
        dataset: list of Dicts with keys: 'sentence', 'paragraph' (has the sentence replaces with [MASK])
    '''
    outputs = []
    for i in range(len(dataset)):
        prompt = f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
        """
        response = get_completion(prompt)
        outputs.append(response)
    return outputs
