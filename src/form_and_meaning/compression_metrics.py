from collections.abc import Sequence

import numpy as np
import pandas as pd

ENGLISH_LANGUAGE = "eng_Latn"


def bits_per_english_character(total_bits: float, english_characters: int) -> float:
    if english_characters <= 0:
        raise ValueError("english_characters must be positive")
    return total_bits / english_characters


def information_parity(english_total_bits: float, target_total_bits: float) -> float:
    if target_total_bits <= 0:
        raise ValueError("target_total_bits must be positive")
    return english_total_bits / target_total_bits


def add_parallel_compression_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("mode", "corpus_key", "checkpoint"),
    language_column: str = "language",
    english_language: str = ENGLISH_LANGUAGE,
) -> pd.DataFrame:
    required = {
        *group_columns,
        language_column,
        "BPC",
        "Total_Characters",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    derived_columns = {
        "Total_Bits",
        "English_Total_Bits",
        "English_Total_Characters",
        "BPEC",
        "IP",
    }
    result = result.drop(columns=derived_columns.intersection(result.columns))
    result["Total_Bits"] = result["BPC"] * result["Total_Characters"]

    english = result[result[language_column] == english_language]
    duplicate_english = english.duplicated(list(group_columns), keep=False)
    if duplicate_english.any():
        duplicate_groups = english.loc[duplicate_english, list(group_columns)].drop_duplicates()
        raise ValueError(
            "Expected one English row per group; duplicates found for:\n"
            f"{duplicate_groups.to_string(index=False)}"
        )

    english = english[
        [*group_columns, "Total_Bits", "Total_Characters"]
    ].rename(
        columns={
            "Total_Bits": "English_Total_Bits",
            "Total_Characters": "English_Total_Characters",
        }
    )
    result = result.merge(english, on=list(group_columns), how="left", validate="many_to_one")

    valid = (
        np.isfinite(result["Total_Bits"])
        & (result["Total_Bits"] > 0)
        & np.isfinite(result["English_Total_Bits"])
        & (result["English_Total_Bits"] > 0)
        & (result["English_Total_Characters"] > 0)
    )
    result["BPEC"] = np.nan
    result["IP"] = np.nan
    result.loc[valid, "BPEC"] = (
        result.loc[valid, "Total_Bits"] / result.loc[valid, "English_Total_Characters"]
    )
    result.loc[valid, "IP"] = (
        result.loc[valid, "English_Total_Bits"] / result.loc[valid, "Total_Bits"]
    )

    return result
