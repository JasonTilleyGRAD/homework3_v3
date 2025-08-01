from .base_llm import BaseLLM
from .sft import test_model
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def tokenize(tokenizer, question: str, answer: str, reasoning: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{reasoning}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str, reasoning: str) -> dict[str, str,str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    

    return {"question": prompt, "answer": f"<answer>{answer}</answer>", "reasoning": reasoning}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(output_dir: str,**kwargs):
    from peft import get_peft_model, LoraConfig
    from transformers import Trainer, TrainingArguments, AutoTokenizer
    from .datagen import generate_dataset

    generate_dataset("data/rft.json")
    
    model = BaseLLM().model
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    rank = 6

    pft_dict = {
        'task_type': 'CAUSAL_LM',
        'target_modules': 'all-linear',
        'bias': 'none',
        'r': rank,
        'lora_alpha': 5*rank,
        # 'lora_dropout': 0.01,
    }

    pft_config = LoraConfig(**pft_dict)
    
    model = get_peft_model(model,pft_config)

    try:
        model.enable_input_require_grads()
    except AttributeError:
        pass

    train_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=1e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=30,
        per_device_train_batch_size=32,    
    )
    
    
    trainset = Dataset("rft")

    
    tokenized_trainset = TokenizedDataset(tokenizer, trainset, format_example)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_trainset,
    )

    trainer.train()

    trainer.save_model(output_dir)




if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
