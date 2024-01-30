import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Bloomz model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")
model.to("cuda")  # Move model to GPU
batch_size = 32

# Rest of your code remains the same

def generate_text(prompts, max_length=512):
    # Tokenize all prompts (batch processing)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")  # Move inputs to GPU

    # Generate responses for each prompt
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)

    # Decode and return the generated text for each input
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Define a function to generate a completion for a given dataset
def BloomzModel(dataset):
    outputs = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        # print(batch)
        prompts = [f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {batch['sentence'][m]}. The paragraph is: {batch['paragraph'][m]}
        """ for m in range(len(batch))]
        # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # outputs.extend(model.generate(inputs))
        print(len(prompts))
        out = generate_text(prompts)
        print(len(out))
        outputs.extend(out)
        print(f"blooms {i} of {len(dataset)} done.", end='\r')
        assert len(outputs) == i + len(batch), f"Expected {i + len(batch)} outputs, got {len(outputs)}"
    return outputs
  