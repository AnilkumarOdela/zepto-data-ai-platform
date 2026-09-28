import sqlite3
import pandas as pd


DATABASE_FILE = "data_pipeline/books.db"
CLEAN_DATA_FILE = "data_pipeline/clean_books.csv"


def create_database():

    # Connect to SQLite database
    connection = sqlite3.connect(DATABASE_FILE)

    cursor = connection.cursor()

    # Enable foreign-key constraints
    cursor.execute("PRAGMA foreign_keys = ON")

    # Create categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Create books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL UNIQUE,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    connection.commit()

    return connection


def load_data(connection):

    # Read cleaned CSV
    df = pd.read_csv(CLEAN_DATA_FILE)

    # Insert categories first
    categories = df["category"].dropna().unique()

    for category in categories:

        connection.execute(
            """
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    connection.commit()

    # Insert books
    for _, row in df.iterrows():

        category_id = connection.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        ).fetchone()[0]

        connection.execute(
            """
            INSERT OR IGNORE INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                category_id
            )
        )

    connection.commit()

    print("Data loaded into SQLite.")


def run_sql_queries(connection):

    print("\n" + "=" * 60)
    print("SQL QUERY RESULTS")
    print("=" * 60)

    # Query 1: SELECT + WHERE
    query1 = """
        SELECT title, price_inr
        FROM books
        WHERE price_inr > 5000
    """

    print("\n1. Books with price above ₹5000:")
    print(pd.read_sql(query1, connection).head())

    # Query 2: ORDER BY
    query2 = """
        SELECT title, rating
        FROM books
        ORDER BY rating DESC
    """

    print("\n2. Books ordered by rating:")
    print(pd.read_sql(query2, connection).head())

    # Query 3: LIMIT
    query3 = """
        SELECT title, price_inr
        FROM books
        LIMIT 10
    """

    print("\n3. First 10 books:")
    print(pd.read_sql(query3, connection))

    # Query 4: DISTINCT
    query4 = """
        SELECT DISTINCT category_name
        FROM categories
    """

    print("\n4. Distinct categories:")
    print(pd.read_sql(query4, connection))

    # Query 5: IN / BETWEEN
    query5 = """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
    """

    print("\n5. Books with rating 4 or 5:")
    print(pd.read_sql(query5, connection).head())

    # Query 6: JOIN
    query6 = """
        SELECT
            b.title,
            b.price_inr,
            b.rating,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
    """

    print("\n6. Books joined with categories:")
    join_result = pd.read_sql(query6, connection)

    print(join_result.head())

    return join_result


def validate_with_pandas(connection):

    print("\n" + "=" * 60)
    print("PANDAS VALIDATION")
    print("=" * 60)

    # Read books from database
    books_df = pd.read_sql(
        "SELECT * FROM books",
        connection
    )

    # Read categories from database
    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        connection
    )

    # Reproduce JOIN using pandas merge
    merged_df = books_df.merge(
        categories_df,
        on="category_id",
        how="inner"
    )

    print("\nPandas merge result:")
    print(
        merged_df[
            [
                "title",
                "price_inr",
                "rating",
                "category_name"
            ]
        ].head()
    )

    print(
        f"\nRows from database: {len(books_df)}"
    )

    print(
        f"Rows after pandas merge: {len(merged_df)}"
    )

    if len(books_df) == len(merged_df):
        print(
            "Validation successful: "
            "JOIN and pandas merge have matching row counts."
        )
    else:
        print(
            "Validation warning: "
            "row counts do not match."
        )


def main():

    connection = create_database()

    try:

        load_data(connection)

        run_sql_queries(connection)

        validate_with_pandas(connection)

    finally:

        connection.close()

    print(
        f"\nDatabase created at: {DATABASE_FILE}"
    )


if __name__ == "__main__":
    main()