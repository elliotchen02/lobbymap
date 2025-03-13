import logging
import sys
import pandas as pd

LOGGING_LEVEL = logging.DEBUG

def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s |  %(levelname)s: %(message)s')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(LOGGING_LEVEL)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def parse_entry(entry: dict, exclude_columns: list, max_text_length: int = 150) -> str:
    """
    Parse an entry into a string that can be fed to an LLM for annotation.
    The entry is a dictionary where the keys are the column names and the values are the column values.
    Checks for a maximum text length and truncates if necessary.

    Args:
        entry: A dictionary where the keys are the column names and the values are the column values of an entry row.
        exclude_columns: A list of column names to exclude from the parsed text.
        max_text_length: The maximum length of the parsed text in characters.
    Returns:
        A string that can be fed to an LLM for annotation.
    """
    parsed_text_array = []
    for key, value in entry.items():
        if key not in exclude_columns:
            if key == "bill_summary_text" and len(value) > max_text_length:
                value = value[:137] + "..."
            parsed_text_array.append(f"{key}: {value}")
    return "\n".join(parsed_text_array)


def save_annotation_to_df(df: pd.DataFrame, annotation_dict: dict):
    """
    Save an LLM annotation to a pandas DataFrame. Appends the annotation to the end of the DataFrame.

    Args:
        df: A pandas DataFrame to save the annotation to.
        annotation_dict: A dictionary where the keys are the column names and the values are the column values of the annotation.
    """
    df.loc[len(df)] = annotation_dict



