import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

FLORES_DATASET = "openlanguagedata/flores_plus"
FLORES_LANGUAGE_ALIASES = {"lvs_Latn": "lat_Latn"}


def checkpoint_paths(root: Path) -> list[Path]:
    paths = [path for path in root.glob("checkpoint-*") if path.is_dir()]
    return sorted(paths, key=lambda path: int(path.name.removeprefix("checkpoint-")))


def load_flores(language: str) -> list[str]:
    dataset = load_dataset(FLORES_DATASET, FLORES_LANGUAGE_ALIASES.get(language, language))
    texts = []
    for split in ("dev", "devtest"):
        if split in dataset:
            texts.extend(text.strip() for text in dataset[split]["text"] if text.strip())
    return texts


def dataloader(texts: list[str], tokenizer, batch_size: int = 4) -> DataLoader:
    encoded = [tokenizer(text, truncation=False) for text in texts]
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(encoded, batch_size=batch_size, shuffle=False, collate_fn=collator)


def batch_statistics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)

    losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    ).masked_fill(~mask, 0)
    true_logits = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    ranks = (shift_logits > true_logits.unsqueeze(-1)).sum(dim=-1) + 1
    reciprocal_ranks = (1.0 / ranks).masked_fill(~mask, 0)
    return losses, reciprocal_ranks, mask


def load_model(checkpoint: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
    model.eval()
    return model, device


def aggregate_metrics(model, device, tokenizer, texts: list[str]) -> dict:
    total_loss = 0.0
    total_reciprocal_rank = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for batch in dataloader(texts, tokenizer):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            losses, reciprocal_ranks, mask = batch_statistics(outputs.logits, labels)
            total_loss += losses.sum().item()
            total_reciprocal_rank += reciprocal_ranks.sum().item()
            total_tokens += mask.sum().item()

    nll = total_loss / total_tokens
    total_characters = sum(len(text) for text in texts)
    return {
        "NLL": nll,
        "Perplexity": math.exp(nll),
        "BPC": total_loss / math.log(2) / total_characters,
        "MRR": total_reciprocal_rank / total_tokens,
        "Total_Valid_Tokens": total_tokens,
        "Total_Characters": total_characters,
    }


def sentence_metrics(model, device, tokenizer, texts: list[str]) -> list[dict]:
    records = []
    offset = 0

    with torch.inference_mode():
        for batch in dataloader(texts, tokenizer):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            losses, reciprocal_ranks, mask = batch_statistics(outputs.logits, labels)

            for row in range(mask.shape[0]):
                token_count = mask[row].sum().item()
                loss = losses[row].sum().item()
                nll = loss / token_count
                text = texts[offset]
                records.append(
                    {
                        "id": offset,
                        "token_length": token_count + 1,
                        "char_length": len(text),
                        "NLL": nll,
                        "Perplexity": math.exp(nll),
                        "BPC": loss / math.log(2) / len(text),
                        "MRR": reciprocal_ranks[row].sum().item() / token_count,
                    }
                )
                offset += 1

    return records


def evaluate_flores(
    checkpoint_root: str,
    tokenizer_dir: str,
    languages: tuple[str, ...],
    mode: str,
    corpus: str,
    output_file: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    texts = {language: load_flores(language) for language in languages}
    records = []

    for checkpoint in checkpoint_paths(Path(checkpoint_root)):
        model, device = load_model(checkpoint)
        for language in languages:
            record = aggregate_metrics(model, device, tokenizer, texts[language])
            record.update(
                {
                    "mode": mode,
                    "language": language,
                    "corpus_key": corpus,
                    "checkpoint": checkpoint.name,
                }
            )
            records.append(record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=4))


def evaluate_paraphrases(
    checkpoint_root: str,
    tokenizer_dir: str,
    text_files: tuple[tuple[str, str], ...],
    mode: str,
    corpus: str,
    checkpoint_label: str,
    output_dir: str,
) -> None:
    checkpoint = checkpoint_paths(Path(checkpoint_root))[-1]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model, device = load_model(checkpoint)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for language, text_file in text_files:
        texts = Path(text_file).read_text().splitlines()
        records = sentence_metrics(model, device, tokenizer, texts)
        for record in records:
            record.update(
                {
                    "mode": mode,
                    "language": language,
                    "corpus_key": corpus,
                    "checkpoint": checkpoint_label,
                }
            )
        (output / f"{corpus}_{mode}_{language}.json").write_text(json.dumps(records, indent=4))
