import logging
import argparse
import sys

LOGGING_LEVEL = logging.INFO

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

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='llm-annotation')
    parser.add_argument('output_path', type=str)
    parser.add_argument('--query', type=str, default='default', help='PostgreSQL query to run.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use for annotation. Default is gpt-4o-mini.')
    parser.add_argument('--prompt', type=str, default='default', 
                        help='Prompt to feed to LLM for annotation. It is one of the following values (but you can add others in params.py): (default, default_with_reasoning, kaist)')
    parser.add_argument('-s', '--annotations_per_save', type=int, default=10000, help='Number of annotations to save to each output csv file. Defaults to 10,000.')
    parser.add_argument('-n','--number_of_annotations', type=int, default=10000, help='Number of total annotations expected to run. Used for progress bar. Defaults to 10,000.')
    parser.add_argument('--max_threads', type=int, default=10, help='The maximum number of threads to use. Defaults to 10.')
    parser.add_argument('--temperature', type=int, default=0, help='Temperature of the model to use. Defaults to 0.')
    parser.add_argument('-m', '--maplight', action='store_true', help='Boolean flag indicating if data being processed comes from Maplight dataset.')
    parser.add_argument('-r', '--reasoning', action='store_true', help='Boolean flag indicating if LLM output should include its reasoning.')
    return parser

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
        if not key or not value:
            continue
        if key not in exclude_columns:
            # Shorten bill summary text to avoid token limit overflow when calling API
            if key == "bill_summary_text" and len(value) > max_text_length:
                value = value[:137] + "..."
            parsed_text_array.append(f"{key}: {value}")
    return "\n".join(parsed_text_array)



