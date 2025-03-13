from enum import Enum
import pandas as pd
from pydantic import BaseModel


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

class AnnotationTable:
    def __init__(self, columns: list[str]):
        """
        Initialize the AnnotationTable object. An AnnotationTable stores information about completed annotations.
        It is designed to eventually be saved to a csv file.
        In addition to storing annotation information, it keeps track of the number of current annotations
        in the table and the number of times it has been saved as a csv file.

        Args:
            columns: A list of the column names of the table.
        """
        self.number_of_annotations = 0
        self.number_of_files = 0
        self.entries = {column: [] for column in columns}

    def __len__(self):
        return self.number_of_annotations
    
    def add_entry(self, entry: dict):
        """
        Add an entry to the annotation table.

        Args:
            entry: A dictionary where the keys are the column names and the values are the column values of an entry row.
        """
        self.number_of_annotations += 1
        for column, value in entry.items():
            self.entries[column].append(value)
    
    def clear_entries(self):
        """
        Clear the table of entries. Does not effect the number of times the table has been saved as a file.
        """
        self.number_of_annotations = 0
        self.entries = {column: [] for column in self.entries.keys()}
    
    def get_file_num(self):
        return self.number_of_files

    def save_to_csv(self, output_path: str):
        """
        Save the annotation table to a csv file.

        Args:
            output_path: The path to the directory where the csv file will be saved.
        """
        file_path = f'{output_path}/annotations_{self.number_of_files}.csv'
        df = pd.DataFrame.from_dict(self.entries)
        df.to_csv(file_path, index=False)
        self.number_of_files += 1