import pandas as pd


GBP_TO_INR = 105.50


def clean_books(input_file, output_file):

    # Load raw data
    df = pd.read_csv(input_file)

    print("Raw data loaded.")
    print(f"Rows: {len(df)}")

    # Remove duplicate books
    df = df.drop_duplicates(
        subset=["title"]
    ).copy()

    # Clean GBP price
    df["price_gbp"] = (
        df["price_gbp"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Convert GBP to INR
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # Convert rating text to numbers
    rating_mapping = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = df["star_rating"].map(
        rating_mapping
    )

    # Convert availability to Boolean
    df["in_stock"] = (
        df["availability"]
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    # Handle invalid prices
    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df[
            "price_gbp"
        ].fillna(median_price)

        df["price_inr"] = (
            df["price_gbp"] * GBP_TO_INR
        ).round(2)

    # Drop rows with invalid ratings
    df = df.dropna(
        subset=["rating"]
    ).copy()

    df["rating"] = df[
        "rating"
    ].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    # Final columns
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    # Save cleaned data
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCleaning completed.")
    print(f"Clean rows: {len(df)}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 cleaned records:")
    print(df.head())

    return df


if __name__ == "__main__":

    clean_books(
        "data_pipeline/raw_books.csv",
        "data_pipeline/clean_books.csv"
    )