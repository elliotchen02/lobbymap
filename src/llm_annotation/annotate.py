import os
from enum import Enum

from openai import OpenAI, ChatCompletion
from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
import psycopg2 
from dotenv import load_dotenv

# Load environment vars from .env
load_dotenv()

lobbyview = psycopg2.connect(
    host=os.environ["LOBBYVIEW_HOST"],
    database=os.environ["LOBBYVIEW_DB"],
    user=os.environ["LOBBYVIEW_USER"],
    password=os.environ["LOBBYVIEW_PASSWORD"]
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = """
Classify the given text which explicitly reports lobbying activities for a bill 
into one of the five types: 'Support', 'Oppose', 'Amend', 'Monitor', or 'Unsure'. 
The text is formatted in a data table containing information about its bill number
and underlying clients. Provide a one-sentence reason for the class you choose.
"""


class PredictedClass(Enum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    AMEND = "AMEND"
    MONITOR = "MONITOR"
    UNSURE = "UNSURE"
    # Define __repr__ otherwise print() displays object memory address
    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self._name_)


class Annotation(BaseModel):
    """Annotation format to be returned by GPT model"""
    filing_uuid: str
    # issue_text: str
    predicted_class: PredictedClass
    reasoning: str


def annotate(data: str) -> Annotation:
    """
    Using OpenAI's API, annotate (classify) lobbying report as either:
        {SUPPORT, OPPOSE, AMEND, MONITOR, UNSURE}
    with respect to the proposed bill at hand.
    
    Args:
        data: cleaned text describing features of desired lobbying bill
    Returns:
        A ParsedChatCompletion object in the form of an Annotation object for the given lobbying report
    Raises:
        Exception if model refuses to provide an answer, possibly for token limit or safety reasons
    """
    chat_completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        store=False,
        messages=[
            {"role": "user", 
                "content": PROMPT},
            {"role": "user",
                "content": data},
            ],
        response_format=Annotation,
        temperature=0,
    )
    annotation: Annotation = chat_completion.choices[0].message
    if annotation.refusal:
        raise Exception("Refusal raised!")
    else:
        return annotation.parsed


def save_annotation_to_df(df: pd.DataFrame, annotation: Annotation):
    row_to_append = pd.DataFrame({
        'filing_uuid': [annotation.filing_uuid],
        #'issue_text': [chat_completion.issue_text],
        'predicted_class': [annotation.predicted_class],
        'reasoning': [annotation.reasoning]
    })
    df = pd.concat([df, row_to_append])
    return df


def main():
    query = "SELECT * FROM relational___lda.filings WHERE filing_year = 2010"  #TODO
    completed_annotations = pd.DataFrame()
    with lobbyview.cursor() as cursor:
        cursor.execute(query)
        column_names = [desc[0] for desc in cursor.description]
        entry: tuple[str] = cursor.fetchone()
        parsed_entry_dict = {col: e for col, e in zip(column_names, entry)}
        # for entry in tqdm(cursor):
        annotation = annotate(str(parsed_entry_dict))
        print("annotation", annotation)
        completed_annotations = save_annotation_to_df(completed_annotations, annotation)
        completed_annotations.to_csv('annotations.csv', index=False)

if __name__ == '__main__':
    main()


