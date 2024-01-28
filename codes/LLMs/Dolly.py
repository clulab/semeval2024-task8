import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# Define a function to generate a completion
def get_completion(prompt):
    outputs = generate_text(prompt)
    return outputs[0]['generated_text']

def DollyModel(dataset):
    outputs = []
    for i in range(0, len(dataset), 32):
        prompts = [f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
        """ for i in range(i, i + 32)]
        responses = generate_text(prompts)
        outputs.extend(responses)
    return outputs
  
