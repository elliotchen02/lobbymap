##########################################################
######### DEFAULT PARAMS, QUERIES, AND PROMPTS ###########
##########################################################

#### MODEL & SCRIPT PARAMS ####
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_OUTPUT_COLUMNS = [
    'filing_uuid',
    'issue_text',
    'predicted_class',
]


#### PROMPTS ####
KAIST_PROMPT = """ 
Classify the given text that explicitly describes lobbying activities for a bill 
into one of the five types: 
'Support', 'Oppose', 'Amend', 'Monitor', or 'Unsure' without explanation.
"""
DEFAULT_PROMPT_WITH_REASONING = """
You are given a description of a lobbying report. Based on the report, 
classify the lobbyist's intentions as one of five categories:
'Support', 'Oppose', 'Amend', 'Monitor', or 'Unsure' if you cannot determine. 
Provide a one-sentence reason for the class you choose.
"""
DEFAULT_PROMPT = """
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
    b.official_title as bill_title,
    b.summary_text as bill_summary_text,
    c.client_name
FROM
    link___lda__congress._issue_paragraphs__bills_updated i
JOIN
    relational___congress.bills b ON i.bill_id = b.bill_id
JOIN
    relational___lda.filings f USING (filing_uuid)
JOIN
    relational___lda.clients c USING (lob_id)
ORDER BY
    f.filing_uuid;
"""

MAPLIGHT_QUERY = """
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

TEST_QUERY = """
SELECT
    f.filing_uuid,
    b.bill_id,
    b.bill_number,
    i.section_id,
    i.paragraph_text as issue_text,
    b.official_title as bill_title,
    b.summary_text as bill_summary_text,
    c.client_name
FROM
    link___lda__congress._issue_paragraphs__bills_updated i
JOIN
    relational___congress.bills b ON i.bill_id = b.bill_id
JOIN
    relational___lda.filings f USING (filing_uuid)
JOIN
    relational___lda.clients c USING (lob_id)
WHERE
    f.filing_uuid IN ('259256d2-9cda-41a9-85c3-8957eada51c6',
        '278ae998-a6f2-4408-aef5-3a1b6897a789')
    AND i.paragraph_text IN (
        'H.R. 3245, the Fairness in Cocaine Sentencing Act, which would remove references to cocaine base from the U.S. Code, thereby greatly reducing the sentences of offenders convicted for offenses involving crack cocaine',
        'Urging reform of disparities in federal penalties for crack cocaine and powdered cocaine offenses - H.R. 3245 & H.R. 1459.'
    )
ORDER BY
    f.filing_uuid;
"""

# Columns from Maplight ground truth to exclude from LLM
DEFAULT_EXCLUDE_COLUMNS = [
    'bill_number',
    'organization_name',
    'disposition',
]