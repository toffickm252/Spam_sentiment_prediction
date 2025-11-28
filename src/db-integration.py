import sqlite3
import pandas as pd
from pathlib import Path

# Paths
CSV_PATH = r"C:\Users\Surface\OneDrive\Documentos\GitHub\Spam_sentiment_prediction\data\Enron.csv"
DATA_FOLDER = Path(r"C:\Users\Surface\OneDrive\Documentos\GitHub\Spam_sentiment_prediction\data")
DB_NAME = "nlp_project.db"
TABLE_NAME = "messages"


def create_and_load_db(csv_path: str, db_path: Path, table_name: str):
    conn = None

    try:
        # Load CSV
        data = pd.read_csv(csv_path)

        # Rename columns — edit if needed
        data.columns = ['message_subject', "message_text", "spam_label"]

        # Select only the columns we need for the database
        data = data[['message_subject', 'message_text', 'spam_label']]

        # Ensure DB folder exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to DB
        conn = sqlite3.connect(str(db_path))

        # Create table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_text TEXT NOT NULL,
                spam_label INTEGER NOT NULL
            );
        """)

        # Insert data
        data.to_sql(table_name, conn, if_exists="replace", index=False) # data is added up not replaced

        print(f"Loaded {len(data)} rows into '{table_name}'.")
        print(f"Database path: {db_path}")

    except FileNotFoundError:
        print(f"CSV not found: {csv_path}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    db_path = DATA_FOLDER / DB_NAME
    create_and_load_db(CSV_PATH, db_path, TABLE_NAME)
