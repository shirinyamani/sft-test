from trl import SFTTrainer
from datasets import load_dataset


if __name__ == "__main__":
    train_dataset = load_dataset("trl-lib/tldr", split="train[:11%]")
    model_id = "Qwen/Qwen2.5-0.5B"
    trainer = SFTTrainer(
        model=model_id,
        train_dataset=train_dataset,
    )
    trainer.train()



