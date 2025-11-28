import sqlite3
from pathlib import Path

# Paths
db_path = Path(r"C:\Users\Surface\OneDrive\Documentos\GitHub\Spam_sentiment_prediction\data\nlp_project.db")
sql_path = Path(r"C:\Users\Surface\OneDrive\Documentos\GitHub\Spam_sentiment_prediction\src\data-cleaning.sql")

# Verify files exist
if not db_path.exists():
    print(f"Error: Database file not found at {db_path}")
    exit(1)

if not sql_path.exists():
    print(f"Error: SQL file not found at {sql_path}")
    exit(1)

try:
    # Connect
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Connected to database: {db_path}")
    
    # Load and run SQL
    with sql_path.open("r", encoding="utf-8") as f:
        script = f.read()
    
    print(f"Executing SQL script from: {sql_path}")
    cursor.executescript(script)
    
    # Finish
    conn.commit()
    print("✓ SQL script executed successfully!")
    print("✓ Changes committed to database.")
    
except sqlite3.Error as e:
    print(f"SQLite error: {e}")
    conn.rollback()
    
except Exception as e:
    print(f"Error: {e}")
    
finally:
    if conn:
        conn.close()
        print("Database connection closed.")