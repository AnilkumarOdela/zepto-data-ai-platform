import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
TARGET_BOOKS = 60


def scrape_books():
    books = []
    page_number = 1

    while len(books) < TARGET_BOOKS:

        # Build listing page URL
        if page_number == 1:
            page_url = BASE_URL
        else:
            page_url = urljoin(
                BASE_URL,
                f"catalogue/page-{page_number}.html"
            )

        print(f"Scraping page {page_number}: {page_url}")

        # Request listing page
        response = requests.get(
            page_url,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Find all books on the page
        book_cards = soup.select(
            "article.product_pod"
        )

        for book in book_cards:

            # -------------------------
            # Title
            # -------------------------
            title = book.h3.a.get("title")

            # -------------------------
            # Price
            # -------------------------
            price_gbp = book.select_one(
                ".price_color"
            ).get_text(strip=True)

            # -------------------------
            # Star rating
            # -------------------------
            rating_element = book.select_one(
                "p.star-rating"
            )

            star_rating = rating_element.get(
                "class"
            )[1]

            # -------------------------
            # Availability
            # -------------------------
            availability = book.select_one(
                ".availability"
            ).get_text(" ", strip=True)

            # -------------------------
            # Book detail URL
            # -------------------------
            relative_url = book.h3.a.get("href")

            detail_url = urljoin(
                page_url,
                relative_url
            )

            # -------------------------
            # Request book detail page
            # -------------------------
            detail_response = requests.get(
                detail_url,
                timeout=10
            )

            detail_response.raise_for_status()

            detail_soup = BeautifulSoup(
                detail_response.text,
                "html.parser"
            )

            # -------------------------
            # Category
            # -------------------------
            breadcrumb_items = detail_soup.select(
                "ul.breadcrumb li"
            )

            if len(breadcrumb_items) >= 3:
                category = breadcrumb_items[
                    2
                ].get_text(strip=True)
            else:
                category = "Unknown"

            # -------------------------
            # Store book
            # -------------------------
            books.append({
                "title": title,
                "price_gbp": price_gbp,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            })

            if len(books) >= TARGET_BOOKS:
                break

        page_number += 1

    return pd.DataFrame(books)


def main():

    df = scrape_books()

    print("\n" + "=" * 50)
    print("SCRAPING COMPLETED")
    print("=" * 50)

    print(f"Total books scraped: {len(df)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nCategories found:")
    print(df["category"].value_counts())

    # Save raw dataset
    output_file = "data_pipeline/raw_books.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nRaw data saved to {output_file}"
    )


if __name__ == "__main__":
    main()