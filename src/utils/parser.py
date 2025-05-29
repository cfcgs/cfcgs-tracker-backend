import pandas as pd
import numpy as np
from fastapi import UploadFile
from typing import Literal
from io import BytesIO

from fuzzywuzzy import process, fuzz

from src.cfcgs_tracker.settings import Settings

def normalize_columns_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia as colunas de um DataFrame com base na correspondência fuzzy com as colunas esperadas,
    e converte nomes finais para snake_case com underscore,
    e trata valores nulos
    """
    original_columns = list(df.columns)
    normalized_columns = {}
    lower_originals = [col.strip().lower() for col in original_columns]

    for expected_col in Settings().expected_columns_set:
        match, score = process.extractOne(expected_col, lower_originals, scorer=fuzz.token_sort_ratio)
        if score >= Settings().SIMILARITY_THRESHOLD:
            matched_index = lower_originals.index(match)
            normalized_columns[original_columns[matched_index]] = expected_col
        else:
            raise ValueError(f"Coluna obrigatória '{expected_col}' não encontrada. Melhor correspondência: '{match}' (score: {score})")

    # Renomeia as colunas com base no dicionário fuzzy
    df = df.rename(columns=normalized_columns)

    # Substitui espaços por "_" para compatibilidade com banco
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    df.replace({"-": np.nan, "": np.nan, "n/a": np.nan, "N/A": np.nan, "null": np.nan, "NULL": np.nan, "na": np.nan, "NA": np.nan}, inplace=True)

    df.dropna(how='all', inplace=True)

    return df

def read_file(file: UploadFile, file_type: Literal["csv", "xlsx"]) -> pd.DataFrame:
    """
        Lê o arquivo (csv ou xlsx), normaliza as colunas e retorna o DataFrame.
        """
    content = file.file.read()
    buffer = BytesIO(content)

    if file_type == "csv":
        df = pd.read_csv(buffer)
    elif file_type == "xlsx":
        df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type.")

    df = normalize_columns_fuzzy(df)
    return df

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None