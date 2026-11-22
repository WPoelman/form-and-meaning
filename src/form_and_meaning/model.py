import json
from itertools import chain
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from form_and_meaning import config


def train(
    train_files: tuple[str, ...],
    tokenizer_dir: str,
    output_dir: str,
    vocab_size: int,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config.model_config(vocab_size), indent=2))

    set_seed(config.SEED)
    splits = [
        load_dataset("text", data_files=train_file, split="train").train_test_split(
            test_size=0.1,
            seed=config.SEED,
        )
        for train_file in train_files
    ]
    raw_train = concatenate_datasets([split["train"] for split in splits])
    raw_eval = concatenate_datasets([split["test"] for split in splits])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(output)).float()

    def tokenize(examples: dict[str, list[str]]) -> dict:
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {key: list(chain.from_iterable(examples[key])) for key in examples}
        total_length = len(concatenated[next(iter(concatenated))])
        total_length = (total_length // config.TRAINING_BLOCK_SIZE) * config.TRAINING_BLOCK_SIZE
        grouped = {
            key: [
                values[index : index + config.TRAINING_BLOCK_SIZE]
                for index in range(0, total_length, config.TRAINING_BLOCK_SIZE)
            ]
            for key, values in concatenated.items()
        }
        grouped["labels"] = grouped["input_ids"].copy()
        return grouped

    def prepare(dataset):
        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, num_proc=4)
        return tokenized.map(group_texts, batched=True, num_proc=4)

    train_dataset = prepare(raw_train)
    eval_dataset = prepare(raw_eval)

    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        adam_beta1=config.ADAM_BETA1,
        adam_beta2=config.ADAM_BETA2,
        adam_epsilon=config.ADAM_EPSILON,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": config.MIN_LEARNING_RATE},
        warmup_steps=config.WARMUP_STEPS,
        bf16=True,
        save_strategy="steps",
        save_steps=config.CHECKPOINT_STEPS,
        eval_strategy="steps",
        eval_steps=config.CHECKPOINT_STEPS,
        logging_steps=100,
        report_to="none",
        seed=config.SEED,
        data_seed=config.SEED,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    final_checkpoint = output / f"checkpoint-{trainer.state.global_step}"
    if not final_checkpoint.exists():
        trainer.save_model(final_checkpoint)
    trainer.save_state()
