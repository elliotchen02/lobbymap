import os
from collections import deque, defaultdict
from enum import Enum
import threading

from openai import OpenAI, ChatCompletion
from tqdm import tqdm
import pandas as pd
import psycopg2 
from dotenv import load_dotenv

from annotation import Annotation, AnnotationWithReasoning, AnnotationTable, PredictedClass
from utils import create_logger, parse_entry, create_parser
from params import *

logger = create_logger("annotations")
argparser = create_parser()
args = argparser.parse_args()

OUTPUT_PATH = args.output_path
PROMPT = DEFAULT_PROMPT
if args.prompt == "default_with_reasoning":
    PROMPT = DEFAULT_PROMPT_WITH_REASONING
elif args.prompt == "kaist":
    PROMPT = KAIST_PROMPT
QUERY = DEFAULT_QUERY       # TODO: Add support for other queries
if args.query == "test":
    QUERY = TEST_QUERY
elif args.query == "maplight":
    QUERY = MAPLIGHT_QUERY
MODEL = args.model
TEMPERATURE = args.temperature
ANNOTATIONS_PER_SAVE = args.annotations_per_save
NUMBER_OF_ANNOTATIONS = args.number_of_annotations
MAX_THREADS = args.max_threads
USING_MAPLIGHT = args.maplight
USING_REASONING = args.reasoning
OUTPUT_COLUMNS = DEFAULT_OUTPUT_COLUMNS
if USING_REASONING:
    OUTPUT_COLUMNS.append('reasoning')
if USING_MAPLIGHT:
    OUTPUT_COLUMNS.extend(['maplight_disposition', 'equals_maplight'])

# Load environment vars from .env
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class AnnotationThread(threading.Thread):
    def __init__(
            self, 
            entry_dict: dict, 
            annotation_count: int, 
            completed_annotations: AnnotationTable,
            df_lock: threading.Lock
        ):
        """
        Initialize the AnnotationThread object. The thread will annotate a single record (entry) via an OpenAI API call
        and save the annotation to a pandas DataFrame that is shared between threads. 

        Args:
            entry_dict: A dictionary where the keys are the column names and the values are the column values of an entry row.
            annotation_count: The index of the annotation being annotated.
            completed_annotations: An AnnotationTable object of completed annotations that is shared between threads.
            df_lock: A threading.Lock object that is used to synchronize access to the completed_annotations DataFrame.
        """
        super().__init__()
        self.entry_dict = entry_dict.copy()
        self.annotation_count = annotation_count
        self.completed_annotations = completed_annotations
        self.df_lock = df_lock

    def save_annotation(self, annotation: Annotation):
        update_dict = {
            'filing_uuid': annotation.filing_uuid,
            'issue_text': annotation.issue_text,
            'predicted_class': annotation.predicted_class,
        }
        if USING_MAPLIGHT:
            # Check if llm annotation matches Maplight ground truth data
            equals_maplight = False
            if str(annotation.predicted_class).lower() == self.entry_dict['disposition'].lower():
                equals_maplight = True
            update_dict['maplight_disposition'] = self.entry_dict['disposition'].upper(),
            update_dict['equals_maplight'] = equals_maplight
        if USING_REASONING:
            update_dict['reasoning'] = annotation.reasoning
        self.completed_annotations.add_entry(update_dict)

    # Override the run method
    def run(self):
        try:
            exclude_columns = DEFAULT_EXCLUDE_COLUMNS if USING_MAPLIGHT else []
            parsed_text = parse_entry(self.entry_dict, exclude_columns)
            llm_annotation = annotate(parsed_text, self.annotation_count)
            # Save annotation to shared completed annotations table
            with self.df_lock:
                self.save_annotation(llm_annotation)
                if len(self.completed_annotations) == ANNOTATIONS_PER_SAVE: 
                    logger.debug(f"Saving file {self.completed_annotations.get_file_num()}!")
                    self.completed_annotations.save_to_csv(OUTPUT_PATH)
                    self.completed_annotations.clear_entries()
        except Exception as e:
            logger.error(f"Annotation {self.annotation_count} with data {self.entry_dict} failed with the following error:\n {e}")

def create_db_connection():
    connection = psycopg2.connect(
        host=os.environ["LOBBYVIEW_HOST"],
        database=os.environ["LOBBYVIEW_DB"],
        user=os.environ["LOBBYVIEW_USER"],
        password=os.environ["LOBBYVIEW_PASSWORD"],
    )
    return connection


def annotate(data: str, annotation_count: int) -> Annotation|AnnotationWithReasoning:
    """
    Using OpenAI's API, annotate (classify) lobbying report as either:
        {SUPPORT, OPPOSE, AMEND, MONITOR, UNSURE}
    with respect to the proposed bill at hand.
    
    Args:
        data: formatted text describing features of desired lobbying bill
    Returns:
        A ParsedChatCompletion object in the form of an Annotation object for the given lobbying report
    Raises:
        Exception if model refuses to provide an answer, possibly for token limit or safety reasons
    """
    response_format = AnnotationWithReasoning if USING_REASONING else Annotation
    chat_completion = client.beta.chat.completions.parse(
        model=MODEL,
        store=False,
        messages=[
            {"role": "user", 
                "content": PROMPT},
            {"role": "user",
                "content": data},
            ],
        response_format=response_format,
        temperature=TEMPERATURE,
    )
    annotation: Annotation|AnnotationWithReasoning = chat_completion.choices[0].message
    if annotation.refusal:
        raise Exception(f"Refusal raised for annotation {annotation_count} with the following data entry:\n {data}")
    else:
        return annotation.parsed


if __name__ == '__main__':
    # Establish DB connection
    lobbyview = create_db_connection()
    logger.info("Connected to database!\n")
    # Dictionary to store annotations that will eventually save to dataframe
    llm_annotations = AnnotationTable(columns=OUTPUT_COLUMNS)
    with lobbyview.cursor() as cursor:
        logger.info("Executing query...\n")
        cursor.execute("SET statement_timeout = %s", ('1000000',))  # timeout in milliseconds
        cursor.execute(QUERY, (NUMBER_OF_ANNOTATIONS, ))
        logger.info("Query returned sucessfully!\n")

        # Create a lock to ensure threads don't overwrite each other on the dataframe
        df_lock = threading.Lock()
        # Maintain a queue of current running threads to limit concurrency
        curr_running_threads = deque()
        # Current annotation number
        annotation_count = 0
        for entry in tqdm(cursor, total=NUMBER_OF_ANNOTATIONS):
            column_names = [desc[0] for desc in cursor.description]
            entry_dict_to_parse = {col: e for col, e in zip(column_names, entry)}

            # If too many threads are running, wait for the oldest thread to finish
            if len(curr_running_threads) >= MAX_THREADS:
                curr_running_threads.popleft().join()

            try:
                thread = AnnotationThread(entry_dict_to_parse, annotation_count, llm_annotations, df_lock)
                curr_running_threads.append(thread)
                thread.start()
                annotation_count += 1
            except Exception as e:
                logger.error(f"Annotation {str(annotation_count)} with data {entry_dict_to_parse} failed with the following error: {e}")
                continue

        # Wait for any remaining threads to finish
        for t in curr_running_threads:
            t.join()

        # Save any remaining annotations
        try:
            if len(llm_annotations) > 0:
                llm_annotations.save_to_csv(OUTPUT_PATH)
        except Exception as e:
            logger.error(f"Failed to save remaining {len(llm_annotations)} annotations with the following error:\n {e}")
        logger.info(f"Saved {annotation_count} annotations to {OUTPUT_PATH}!")

