# Data Pipeline

This module implements the data ingestion, cleaning, transformation, and
relational storage pipeline for the Zepto Data & AI Platform.

## Pipeline

Books to Scrape
→ Web Scraping
→ Raw CSV
→ Data Cleaning
→ GBP to INR Conversion
→ SQLite Database
→ SQL Queries
→ Pandas Validation

## Files

- `scraper.py` - Scrapes book information from Books to Scrape.
- `cleaner.py` - Cleans the scraped data and converts GBP prices to INR.
- `database.py` - Creates the normalized SQLite database and runs SQL/Pandas
  validation.
- `raw_books.csv` - Generated raw scraped dataset.
- `clean_books.csv` - Generated cleaned dataset.
- `books.db` - Generated SQLite database.

## Run

From the project root:

```powershell
python data_pipeline\scraper.py
python data_pipeline\cleaner.py
python data_pipeline\database.py