import os

from openai import OpenAI
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

def annotate(data):
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[{"role": "user", "content": "who is the current president in one word?"}],
    )

def main():
    query = "SELECT * FROM relational___lda.filings WHERE filing_year = 2010"  #TODO
    with lobbyview.cursor() as cursor:
        cursor.execute(query)
        print(cursor.fetchone())


        # for entry in cursor:
        #     annotation = annotate(entry)
        #     # TODO update db with annotations

if __name__ == '__main__':
    main()


