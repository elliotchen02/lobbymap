import logging
import pandas as pd


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s |  %(levelname)s: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def parse_entry(entry: dict, exclude_columns: list) -> str:
    """
    Parse a SQL row into a string that can be fed to an LLM for annotation.
    The SQL row is a dictionary where the keys are the column names and the values are the column values.
    Checks for a maximum text length and truncates if necessary.

    Args:
        entry: A dictionary where the keys are the column names and the values are the column values of an entry row.
        exclude_columns: A list of column names to exclude from the parsed text.
    Returns:
        A string that can be fed to an LLM for annotation.
    """
    parsed_text_array = []
    for key, value in entry.items():
        if key not in exclude_columns:
            if key == "bill_summary_text" and len(value) > 140:
                value = value[:137] + "..."
            parsed_text_array.append(f"{key}: {value}")
    return "\n".join(parsed_text_array)


def save_annotation_to_df(df: pd.DataFrame, annotation_dict: dict) -> pd.DataFrame:
    """
    Save an LLM annotation to a pandas DataFrame.

    Args:
        df: A pandas DataFrame to save the annotation to.
        annotation_dict: A dictionary where the keys are the column names and the values are the column values of the annotation.
    Returns:
        A pandas DataFrame with the annotation appended.
    """
    row_to_append = pd.DataFrame({
        column: [value] for column, value in annotation_dict.items()
    })
    df = pd.concat([df, row_to_append])
    return df