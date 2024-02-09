import openai
import pickle
import multiprocessing

# Set OpenAI API key
api_key = "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"
openai.api_key = api_key

def get_completion(prompts, model="davinci-002"):
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,  # The degree of randomness of the model's output
    )
    return response.choices[0].text.strip()

def process_dataset(dataset, start_index, end_index, outputs):
    for i in range(start_index, end_index):
        if i < len(outputs):
            continue
        prompt = f"""
        Replace [MASK] in following paragraph with one sentence that has a meaning similar to: {dataset[i]['sentence']}. The paragraph is: {dataset[i]['paragraph']}
        """
        response = get_completion(prompt)
        outputs.append(response)
        print(f"davinci is at {i} of {end_index}", end='\r')
        if i % 100 == 0:
            pickle.dump(outputs, open("davinciOutputs.pkl", "wb"))

if __name__ == '__main__':
    dataset = [...]  # Your dataset here
    num_processes = 6  # Number of parallel processes
    outputs = []

    # Split the dataset into equal chunks for each process
    chunk_size = len(dataset) // num_processes
    processes = []

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < num_processes - 1 else len(dataset)
        process = multiprocessing.Process(target=process_dataset, args=(dataset, start_index, end_index, outputs))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    pickle.dump(outputs, open("davinciOutputs.pkl", "wb"))
