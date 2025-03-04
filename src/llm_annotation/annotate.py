import os
from enum import Enum

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
PROMPT = PROMPT_1_NO_REASON

# Load environment vars from .env
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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


def run_annotations():
    lobbyview = psycopg2.connect(
        host=os.environ["LOBBYVIEW_HOST"],
        database=os.environ["LOBBYVIEW_DB"],
        user=os.environ["LOBBYVIEW_USER"],
        password=os.environ["LOBBYVIEW_PASSWORD"],
    )
    logger.info("Connected to database!\n")
    completed_annotations = pd.DataFrame()
    with lobbyview.cursor() as cursor:
        logger.info("Executing query...\n")
        # cursor.execute("SET statement_timeout = %s", ('1200000',))
        cursor.execute(DEFAULT_QUERY, (NUMBER_OF_ANNOTATIONS,))
        logger.info("Query returned sucessfully!\n")

        annotation_count = 0
        current_file_count = 0
        for entry in tqdm(cursor, total=NUMBER_OF_ANNOTATIONS):
            column_names = [desc[0] for desc in cursor.description]
            entry_dict_to_parse = {col: e for col, e in zip(column_names, entry)}
            parsed_text = parse_entry(entry_dict_to_parse, EXCLUDE_COLUMNS)
            try:
                llm_annotation = annotate(str(parsed_text), annotation_count)
            except Exception as e:
                logger.error(e, annotation_count)
                continue
            
            # Check if llm annotation matches ground truth
            equals_maplight = False
            if str(llm_annotation.predicted_class).lower() == entry_dict_to_parse['disposition'].lower():
                equals_maplight = True
            
            # Save progress
            llm_annotation_dict = {
                'filing_uuid': llm_annotation.filing_uuid,
                'issue_text': llm_annotation.issue_text,
                'predicted_class': llm_annotation.predicted_class,
                # TODO 'reasoning': llm_annotation.reasoning,
                'equals_maplight': equals_maplight
            }
            completed_annotations = save_annotation_to_df(completed_annotations, llm_annotation_dict)
            if annotation_count % ANNOTATIONS_PER_SAVE == 0 and annotation_count != 0: 
                logger.info(f"Saving file {current_file_count}!")
                completed_annotations.to_csv(f'annotations/annotations_{current_file_count}.csv', Index=False)
                completed_annotations = pd.DataFrame()
                current_file_count += 1
            annotation_count += 1

        # Save any remaining annotations
        completed_annotations.to_csv(f'annotations/annotations_{current_file_count}.csv', Index=False)
        logger.info(f"Saved {annotation_count} annotations to annotations.csv!")
        
if __name__ == '__main__':
    run_annotations()


