import math
import re

import pandas as pd
import numpy as np
from fastapi import UploadFile
from typing import Literal
from io import BytesIO

from fuzzywuzzy import process, fuzz

from src.cfcgs_tracker.settings import Settings


def normalize_columns_fuzzy(
    df: pd.DataFrame, upload_type: int
) -> pd.DataFrame:
    """
    Renomeia as colunas de um DataFrame com base na correspondência fuzzy com as colunas esperadas,
    e converte nomes finais para snake_case com underscore,
    e trata valores nulos
    """
    original_columns = list(df.columns)
    normalized_columns = {}
    lower_originals = [col.strip().lower() for col in original_columns]

    if upload_type == 1:
        for expected_col in Settings().expected_columns_set:
            match, score = process.extractOne(
                expected_col, lower_originals, scorer=fuzz.token_sort_ratio
            )
            if score >= Settings().SIMILARITY_THRESHOLD:
                matched_index = lower_originals.index(match)
                normalized_columns[original_columns[matched_index]] = (
                    expected_col
                )
            else:
                raise ValueError(
                    f"Coluna obrigatória '{expected_col}' não encontrada. Melhor correspondência: '{match}' (score: {score})"
                )

    if upload_type == 3:
        patterns = {
            r"adaptation-related development finance.*commitment.*\d{4} usd thousand": "adaptation-related_development_finance_-_commitment_-_current_usd_thousand",
            r"mitigation-related development finance.*commitment.*\d{4} usd thousand": "mitigation-related_development_finance_-_commitment_-_current_usd_thousand",
        }

        for original_col in original_columns:
            col_lower = original_col.strip().lower()
            for pattern, new_name in patterns.items():
                if re.search(pattern, col_lower):
                    normalized_columns[original_col] = new_name
                    break

    # Renomeia as colunas com base no dicionário fuzzy
    df = df.rename(columns=normalized_columns)

    # Substitui espaços por "_" para compatibilidade com banco
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    df.replace(
        {
            "-": np.nan,
            "": np.nan,
            "n/a": np.nan,
            "N/A": np.nan,
            "null": np.nan,
            "NULL": np.nan,
            "na": np.nan,
            "NA": np.nan,
        },
        inplace=True,
    )

    df.dropna(how="all", inplace=True)

    return df


def read_file(
    file: UploadFile, file_type: Literal["csv", "xlsx"], upload_type: int
) -> pd.DataFrame:
    """
    Lê o arquivo (csv ou xlsx), normaliza as colunas e retorna o DataFrame.
    """
    content = file.file.read()
    buffer = BytesIO(content)

    if file_type == "csv":
        df = pd.read_csv(buffer)
    elif file_type == "xlsx":
        if upload_type == 3:
            df = pd.read_excel(buffer, sheet_name=1)
        elif upload_type == 2:
            df = pd.read_excel(buffer, sheet_name=2)
        else:
            df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type.")

    df = normalize_columns_fuzzy(df, upload_type)

    return df


def safe_float(value):
    try:
        f_value = float(value)
        if math.isnan(f_value):
            return None
        return f_value
    except (ValueError, TypeError):
        return None


def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
