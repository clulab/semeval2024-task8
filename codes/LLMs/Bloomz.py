import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Bloomz model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")

# Define a function to generate a completion
def get_completion(prompt, model):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0])

# Define a function to generate a completion for a given dataset
def BloomzModel(dataset):
    outputs = []
    for i in range(len(dataset)):
        prompt = f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
        """
        response = get_completion(prompt, model)
        outputs.append(response)
    return outputs
  
