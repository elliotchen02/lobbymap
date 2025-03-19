##########################################################
############# Parameters To Be Set By User ###############
##########################################################

#### MODEL & SCRIPT PARAMS ####
ANNOTATIONS_PER_SAVE = 10000    # Number of annotations to save to each CSV (saving progress)
NUMBER_OF_ANNOTATIONS = 250000
MAX_THREADS = 10
MODEL = "gpt-4o-mini"
# Value from 0 to 2, higher = more creative
# See https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
TEMPERATURE = 0
OUTPUT_COLUMNS = [
    'filing_uuid',
    'issue_text',
    'predicted_class',
    # TODO 'reasoning': llm_annotation.reasoning,
    'maplight_disposition',
    'equals_maplight',
]
OUTPUT_PATH = "annotations/maplight_annotations"


#### PROMPTS ####
# Default prompt is prompt no. 0
DEFAULT_PROMPT = """ 
Classify the given text that explicitly describes lobbying activities for a bill 
into one of the five types: 
'Support', 'Oppose', 'Amend', 'Monitor', or 'Unsure' without explanation.
"""
PROMPT_1 = """
You are given a description of a lobbying report. Based on the report, 
classify the lobbyist's intentions as one of five categories:
'Support', 'Oppose', 'Amend', 'Monitor', or 'Unsure' if you cannot determine. 
Provide a one-sentence reason for the class you choose.
"""
PROMPT_1_NO_EXP = """
You are given a description of a lobbying report. Based on the report, 
classify the lobbyist's intentions as one of five categories:
'Support', 'Oppose', 'Amend', 'Monitor', or 'Unsure' if you cannot determine.
Provide no explanation.
"""


#### POSTEGRESQL QUERY PARAMS ####
DEFAULT_QUERY = """
SELECT
    f.filing_uuid,
    b.bill_id,
    b.bill_number,
    i.section_id,
    i.paragraph_text as issue_text,
    c.name,
    b.official_title as bill_title,
    b.summary_text as bill_summary_text,
    m.organization_name,
    m.disposition
FROM
    link___lda__congress._issue_paragraphs__bills_updated i
JOIN
    relational___congress.bills b ON i.bill_id = b.bill_id
JOIN
    relational___lda.filings f USING (filing_uuid)
JOIN
    maplight.disam_2024 c USING (lob_id)
JOIN
    maplight.bill_org_disp_fixed m ON b.bill_id = m.bill_id AND c.organization_id = m.organization_id
ORDER BY 
    f.filing_uuid;
"""

# Columns from Maplight ground truth to exclude from LLM
EXCLUDE_COLUMNS = [
    'bill_number',
    'organization_name',
    'disposition',
]