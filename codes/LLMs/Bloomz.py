import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Bloomz model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")
batch_size = 32

# Define a function to generate a completion
def get_completion(prompt, model):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0])

# Define a function to generate a completion for a given dataset
def BloomzModel(dataset):
    outputs = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {data['sentence']}. The paragraph is: {data['paragraph']}
        """ for data in batch]
        inputs = tokenizer.encode(prompts, return_tensors="pt")
        outputs.extend(model.generate(inputs))
    return outputs
  
