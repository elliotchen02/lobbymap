import os
from collections import deque
from enum import Enum
import threading
from typing import override

from pydantic import BaseModel
from openai import OpenAI, ChatCompletion
from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
import psycopg2 
from dotenv import load_dotenv

from utils import create_logger, save_annotation_to_df, parse_entry
from params import *

logger = create_logger("annotations")

# TODO Select your prompt 
PROMPT = PROMPT_1_NO_EXP

# Load environment vars from .env
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class AnnotationThread(threading.Thread):
    def __init__(
            self, 
            record_dict: dict, 
            annotation_count: int, 
            completed_annotations: pd.DataFrame,
            df_lock: threading.Lock
        ):
        """
        Initialize the AnnotationThread object. The thread will annotate a single record (entry) via an OpenAI API call
        and save the annotation to a pandas DataFrame that is shared between threads. 

        Args:
            record_dict: A dictionary where the keys are the column names and the values are the column values of an entry row.
            annotation_count: The index of the annotation being annotated.
            completed_annotations: A pandas DataFrame that is shared between threads.
            df_lock: A threading.Lock object that is used to synchronize access to the completed_annotations DataFrame.
        """
        super().__init__()
        self.record_dict = record_dict.copy()
        self.parsed_text = parse_entry(self.record_dict, EXCLUDE_COLUMNS)
        self.annotation_count = annotation_count
        self.df = completed_annotations
        self.df_lock = df_lock

    @override
    def run(self):
        # Use of global variables is dangerous and should be avoided, 
        # but in this case, it is necessary to share the completed_annotations DataFrame between threads
        global current_file_count
        try:
            llm_annotation = annotate(self.parsed_text, self.annotation_count)

            # Check if llm annotation matches Maplight ground truth data
            equals_maplight = False
            if str(llm_annotation.predicted_class).lower() == self.record_dict['disposition'].lower():
                equals_maplight = True
            
            # Save annotation to pandas DataFrame
            llm_annotation_dict = {
                'filing_uuid': llm_annotation.filing_uuid,
                'issue_text': llm_annotation.issue_text,
                'predicted_class': llm_annotation.predicted_class,
                # TODO 'reasoning': llm_annotation.reasoning,
                'maplight_disposition': self.record_dict['disposition'].upper(),
                'equals_maplight': equals_maplight
            }
            with self.df_lock:
                save_annotation_to_df(self.df, llm_annotation_dict)
                if len(self.df) % ANNOTATIONS_PER_SAVE == 0 and len(self.df) != 0: 
                    logger.debug(f"Saving file {current_file_count}!")
                    self.df.to_csv(f'{OUTPUT_PATH}/annotations_{current_file_count}.csv',index=False)
                    self.df.drop(self.df.index, inplace=True)
                    current_file_count += 1
        except Exception as e:
            logger.error(e)


class PredictedClass(Enum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    AMEND = "AMEND"
    MONITOR = "MONITOR"
    UNSURE = "UNSURE"
    # Define __repr__ otherwise print() displays object memory address
    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self._name_)

    def __str__(self):
        return self.name


class Annotation(BaseModel):
    """Annotation format to be returned by GPT model"""
    filing_uuid: str
    issue_text: str
    predicted_class: PredictedClass
    #TODO reasoning: str


def create_db_connection():
    connection = psycopg2.connect(
        host=os.environ["LOBBYVIEW_HOST"],
        database=os.environ["LOBBYVIEW_DB"],
        user=os.environ["LOBBYVIEW_USER"],
        password=os.environ["LOBBYVIEW_PASSWORD"],
    )
    return connection


def annotate(data: str, annotation_count: int) -> Annotation:
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
    chat_completion = client.beta.chat.completions.parse(
        model=MODEL,
        store=False,
        messages=[
            {"role": "user", 
                "content": PROMPT},
            {"role": "user",
                "content": data},
            ],
        response_format=Annotation,
        temperature=TEMPERATURE,
    )
    annotation: Annotation = chat_completion.choices[0].message
    if annotation.refusal:
        raise Exception(f"Refusal raised for annotation {annotation_count} with the following data entry:\n {data}")
    else:
        return annotation.parsed



if __name__ == '__main__':
    lobbyview = create_db_connection()
    logger.info("Connected to database!\n")
    completed_annotations = pd.DataFrame(columns=OUTPUT_COLUMNS)
    with lobbyview.cursor() as cursor:
        logger.info("Executing query...\n")
        cursor.execute(DEFAULT_QUERY, (NUMBER_OF_ANNOTATIONS,))
        logger.info("Query returned sucessfully!\n")

        # Create a lock to ensure threads don't overwrite each other on the dataframe
        df_lock = threading.Lock()
        # Maintain a queue of current running threads to limit concurrency
        curr_running_threads = deque()
        # Current annotation number
        annotation_count = 0
        # Current file number (to be saved to csv)
        current_file_count = 0
        for entry in tqdm(cursor, total=NUMBER_OF_ANNOTATIONS):
            column_names = [desc[0] for desc in cursor.description]
            entry_dict_to_parse = {col: e for col, e in zip(column_names, entry)}

            # If too many threads are running, wait for the oldest thread to finish
            if len(curr_running_threads) >= MAX_THREADS:
                curr_running_threads.popleft().join()

            thread = AnnotationThread(entry_dict_to_parse, annotation_count, completed_annotations, df_lock)
            curr_running_threads.append(thread)
            thread.start()
            annotation_count += 1

        # Wait for any remaining threads to finish
        for t in curr_running_threads:
            t.join()

        # Save any remaining annotations
        if len(completed_annotations) > 0:
            completed_annotations.to_csv(f'{OUTPUT_PATH}/annotations_{current_file_count}.csv', index=False)
        logger.info(f"Saved {annotation_count} annotations to {OUTPUT_PATH}!")
        


