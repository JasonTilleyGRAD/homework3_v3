 



def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    from .base_llm import BaseLLM
    import json
    import tqdm
    import os
    import random
    import torch

    dataset = []

    if os.path.exists(output_json): 
        size = 1
        with open(output_json, 'r') as f:
            dataset = json.load(f)

    else:
        size = 50

    model = CoTModel('HuggingFaceTb/SmolLM2-1.7B-instruct')
    trainset = Dataset("train")


 
    sampled = random.sample(list(trainset), k=min(size, len(trainset)))

    question, correct_answer = zip(*sampled)

    prompts = [model.format_prompt(q) for q in question]
    generations = model.batched_generate(prompts,num_return_sequences = 10,temperature = temperature)
   
    for i in range(len(question)):
        q = question[i]
        correct = correct_answer[i]
        gen_list = generations[i]
        gen_list_parsed = [model.parse_answer(g) for g in gen_list]
                
        for j in range(len(gen_list)):
            if is_answer_valid(gen_list_parsed[j], correct):
                dataset.append([q, correct, gen_list[j]])
                print(f"Success on index {i}")
                break


               
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)



if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
