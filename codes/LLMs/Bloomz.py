import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Bloomz model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")
batch_size = 32

# Function to generate a completion
def get_completion(prompt, model):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to generate completions for a given dataset
def BloomzModel(dataset):
    outputs = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [f"""
        Replace [MASK] in the following paragraph with one sentence that has a meaning similar to: {batch['sentence'][m]}. The paragraph is: {batch['paragraph'][m]}
        """ for m in range(len(batch))]

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            batch_outputs = model.generate(inputs)
            outputs.extend([tokenizer.decode(o, skip_special_tokens=True) for o in batch_outputs])
            print(f"blooms {i} of {len(dataset)} done.")
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

        assert len(outputs) == i + len(batch)
    return outputs
