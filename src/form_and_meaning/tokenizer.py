from collections.abc import Iterable
from pathlib import Path

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from form_and_meaning import config


def train_tokenizer(lines: Iterable[str], output_dir: Path, vocab_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = {
        "pad_token": AddedToken("<pad>"),
        "bos_token": AddedToken("<s>"),
        "eos_token": AddedToken("</s>"),
        "unk_token": AddedToken("<unk>"),
    }
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        lines,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(special_tokens.values()),
    )
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special_tokens)
    wrapped.model_max_length = config.MAX_SEQUENCE_LENGTH
    wrapped.model_input_names = ["input_ids", "attention_mask"]
    wrapped.save_pretrained(output_dir)


def train_monolingual(corpus_dir: str, output_root: str) -> None:
    corpus = Path(corpus_dir)
    for text_file in sorted(corpus.glob("*.txt")):
        with text_file.open() as lines:
            train_tokenizer(lines, Path(output_root) / text_file.stem, config.MONO_VOCAB_SIZE)


def train_multilingual(corpus_dir: str, output_dir: str) -> None:
    corpus = Path(corpus_dir)

    def lines() -> Iterable[str]:
        for text_file in sorted(corpus.glob("*.txt")):
            with text_file.open() as handle:
                for index, line in enumerate(handle):
                    if index >= config.MULTI_TOKENIZER_LINES_PER_LANGUAGE:
                        break
                    yield line

    train_tokenizer(lines(), Path(output_dir), config.MULTI_VOCAB_SIZE)
