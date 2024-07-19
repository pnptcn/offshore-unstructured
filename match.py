from numpy import ma
import pandas as pd
import re
from unidecode import unidecode
from neo4j import GraphDatabase
from gliner import GLiNER
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from rich.console import Console
from rich.table import Table
import signal
import sys
import logging
from difflib import SequenceMatcher
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize rich console
console = Console()

# Initialize GLiNER model. This model can do NER, without being restricted to predefined labels.
model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)

# Define labels for entity extraction. For improved accuracy regarding addresses, we attempt
# matching individual address componenets, and later join them together.
labels = ["person", "organization", "street", "postcode", "city", "country"]

# Neo4j connection details
uri = "neo4j://localhost:7687"
user = "neo4j"
password = os.environ.get("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(user, password))

def load_accounts_data(file_path='accounts.csv'):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading accounts data: {e}")
        raise

def preprocess_text(text):
    if not isinstance(text, str):
        logging.warning(f"Non-string input to preprocess_text: {text}")
        return ""
    try:
        # Remove common company or person suffixes and titles. It helps with disambiguation.
        text = re.sub(r'\b(ltd|limited|inc|incorporated|llc|gmbh|ag|sa|nv|bv|dr|mr|mrs|ms)\b', '', text, flags=re.IGNORECASE)
        # Remove periods and commas from the text.
        text = re.sub(r'[.,]', '', text)
        # Normalize and cleanup the text.
        text = unidecode(text.lower().strip())
        return text
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ""

def clean_company_name(name):
    # Remove anything in parentheses at the end of the company name.
    # This is to remove any "formerly" or "incorporated" text.
    # Also helps when an organization includes an abrreviation.
    cleaned_name = re.sub(r'\s*\(.*\)$', '', name).strip()
    return cleaned_name

def extract_entities(text):
    if not isinstance(text, str):
        logging.warning(f"Non-string input to extract_entities: {text}")
        return []
    try:
        # Run GliNER on the input text, using the labels we defined earlier.
        return model.predict_entities(text, labels)
    except Exception as e:
        logging.error(f"Error extracting entities: {e}")
        return []

# To deal with many types of ambiguity that would prevent legitimate matches being found,
# we use very broad strokes to retrieve an initial set of potential matches.
def match_entity_broad(tx, entity_type, field_name, entity_text):
    try:
        split_text = entity_text.split(" ")

        # One word matches return too many false positives, so we ignore them.
        # More often than not, these are company names, and the ones with a single
        # word are quite obvious anyway.
        if len(split_text) <= 1:
            return []

        # We take any matches that contain all of the words in the input text in any order.
        query_parts = " AND ".join([f"(toLower(n.{field_name}) CONTAINS ' {part}' OR toLower(n.{field_name}) CONTAINS '{part} ')" for part in split_text])
        query = f"MATCH (n:{entity_type}) WHERE {query_parts} RETURN n.{field_name} AS name, n.countries AS countries, n.sourceID AS source_id"
        results = tx.run(query)
        return [dict(record) for record in results]
    except Exception as e:
        logging.error(f"Error broad matching entity: {e}")
        return []

# To narrow down the broad search, we calculate the Levenshtein distance between the input text
# and the entity name. We can then filter out any matches that are too far away.
def match_entity(tx, entity_type, field_name, entity_text):
    try:
        broad_matches = match_entity_broad(tx, entity_type, field_name, entity_text)
        matches = []
        for record in broad_matches:
            similarity = SequenceMatcher(None, entity_text, record["name"].lower()).ratio()
            if similarity > 0.0:
                match = dict(record)
                match["score"] = similarity
                matches.append(match)
        return sorted(matches, key=lambda x: x["score"], reverse=True)[:5]
    except Exception as e:
        logging.error(f"Error refining entity match: {e}")
        return []

def process_row(row):
    try:
        name = clean_company_name(row['processed_name'])
        contact_details = str(row['contact_details']) if pd.notna(row['contact_details']) else ""

        matches = []

        # Always try to match the name as an Entity
        with driver.session() as session:
            entity_matches = session.execute_read(match_entity, "Entity", name)

            matches.extend([{
                'account': row['name'],
                'original': name,
                'cleaned': name,
                'type': 'Entity',
                'matched_on': 'name',
                **match
            } for match in entity_matches])

        # Extract and match entities from contact details
        if contact_details:
            entities = extract_entities(contact_details)
            address = construct_address(entities)
            if address:
                with driver.session() as session:
                    address_matches = session.execute_read(match_address, preprocess_text(address))
                    matches.extend([{
                        'account': row['name'],
                        'original': address,
                        'cleaned': preprocess_text(address),
                        'type': 'Address',
                        'matched_on': 'address',
                        **match
                    } for match in address_matches])

            for entity in entities:
                if entity['label'] == 'person':
                    with driver.session() as session:
                        entity_matches = session.execute_read(match_entity, "Officer", preprocess_text(entity['text']))

                        matches.extend([{
                            'account': row['name'],
                            'original': entity['text'],
                            'cleaned': preprocess_text(entity['text']),
                            'type': 'Officer',
                            'matched_on': entity['label'],
                            **match
                        } for match in entity_matches])
                elif entity['label'] == 'organization':
                    cleaned_entity_text = clean_company_name(preprocess_text(entity['text']))

                    with driver.session() as session:
                        entity_matches = session.execute_read(match_entity, "Entity", cleaned_entity_text)

                        matches.extend([{
                            'account': row['name'],
                            'original': entity['text'],
                            'cleaned': cleaned_entity_text,
                            'type': 'Entity',
                            'matched_on': entity['label'],
                            **match
                        } for match in entity_matches])

        return {'name': row['name'], 'matches': matches}
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return {'name': row['name'], 'matches': []}

def log_result(result):
    if not result['matches']:
        return

    table = Table(title=f"Matches for {result['name']}")
    table.add_column("Account", style="green")
    table.add_column("Original", style="cyan")
    table.add_column("Cleaned", style="magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Matched On", style="magenta")
    table.add_column("Matched Name", style="green")
    table.add_column("Countries", style="yellow")
    table.add_column("Source ID", style="blue")
    table.add_column("Score", style="red")

    for match in result['matches']:
        table.add_row(
            match['account'],
            match['original'],
            match['cleaned'],
            match['type'],
            match['matched_on'],
            match['name'],
            match['countries'],
            match['source_id'],
            f"{match['score']:.2f}"
        )

    console.print(table)

def save_results_to_csv(results, filename='matches.csv'):
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['account_name', 'match_type', 'matched_on', 'matched_name', 'countries', 'source_id', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                for match in result['matches']:
                    writer.writerow({
                        'account': match['account'],
                        'original': match['original'],
                        'cleaned': match['cleaned'],
                        'type': match['type'],
                        'matched_on': match['matched_on'],
                        'matched_name': match['matched_name'],
                        'countries': match['countries'],
                        'source_id': match['source_id'],
                        'score': f"{match['score']:.2f}"
                    })
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

def main():
    try:
        # Load and preprocess the data
        df = load_accounts_data()
        df['processed_name'] = df['name'].apply(preprocess_text)

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_row = {executor.submit(process_row, row): row for _, row in df.iterrows()}
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    result = future.result()
                    results.append(result)
                    log_result(result)
                except Exception as exc:
                    logging.error(f'Row {row.name} generated an exception: {exc}')

        save_results_to_csv(results)
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

def signal_handler(sig, frame):
    logging.info('Interrupt received, closing Neo4j connection and exiting...')
    driver.close()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        driver.close()

