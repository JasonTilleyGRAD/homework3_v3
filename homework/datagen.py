 



def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    from .base_llm import BaseLLM
    import json
    import tqdm
    import os
    import random
    import torch

    if not os.path.exists(output_json): 
        model = CoTModel('HuggingFaceTb/SmolLM2-1.7B-instruct')
        trainset = Dataset("train")

        dataset = []
        

        for _ in tqdm.tqdm(range(100)):
            
            question, correct_answer = random.choice(list(trainset))
            prompts = [model.format_prompt(q) for q in question]
            generations = model.batched_generate(prompts,num_return_sequences = oversample,temperature = temperature)
            torch.mps.empty_cache()
            for raw_answer in generations: 
                if is_answer_valid(raw_answer, correct_answer):
                    dataset.append([question, correct_answer, raw_answer])
                    print("Success")
    
                    
        with open(output_json, 'w') as f:
            json.dump(dataset, f, indent=2)



if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
