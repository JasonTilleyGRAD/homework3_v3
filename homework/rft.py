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


def train_model(output_dir: str,**kwargs):
    # Reuse much of the SFT code here
    from peft import get_peft_model, LoraConfig
    from transformers import Trainer, TrainingArguments, AutoTokenizer
    from .sft import TokenizedDataset, format_example
    from .datagen import generate_dataset
    
    model = BaseLLM().model
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    rank = 2

    pft_dict = {
        'task_type': 'CAUSAL_LM',
        'target_modules': 'all-linear',
        'bias': 'none',
        'r': rank,
        'lora_alpha': 4*rank,
    }

    pft_config = LoraConfig(**pft_dict)
    
    model = get_peft_model(model,pft_config)

    try:
        model.enable_input_require_grads()
    except AttributeError:
        pass

    train_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=1e-4,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,    
    )
    generate_dataset('data/rft.json')
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
