import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_FILE = "analytics/titanic.csv"


def load_and_save_titanic():

    df = sns.load_dataset("titanic")

    print("Titanic dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nDataset saved to {OUTPUT_FILE}")

    return df


def analyze_missing_values(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing_count = df.isna().sum()

    missing_percentage = (
        df.isna().mean() * 100
    ).round(2)

    missing_summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    print(
        missing_summary[
            missing_summary["missing_count"] > 0
        ]
    )


def clean_data():

    df = pd.read_csv(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    # Age: 5%–30% missing → median
    if df["age"].isna().any():

        age_median = df["age"].median()

        df["age"] = df["age"].fillna(
            age_median
        )

        print(
            f"Age missing values filled "
            f"using median: {age_median:.2f}"
        )

    # Embarked: <5% missing → mode
    if df["embarked"].isna().any():

        embarked_mode = df["embarked"].mode()[0]

        df["embarked"] = df["embarked"].fillna(
            embarked_mode
        )

        print(
            f"Embarked missing values filled "
            f"using mode: {embarked_mode}"
        )

    # Embark town: <5% missing → mode
    if df["embark_town"].isna().any():

        embark_town_mode = df[
            "embark_town"
        ].mode()[0]

        df["embark_town"] = df[
            "embark_town"
        ].fillna(
            embark_town_mode
        )

        print(
            f"Embark town missing values filled "
            f"using mode: {embark_town_mode}"
        )

    # Deck: >30% missing → encode as Unknown
    if df["deck"].isna().any():

        df["deck"] = df["deck"].fillna(
            "Unknown"
        )

        print(
            "Deck missing values encoded "
            "as 'Unknown'."
        )

    print("\nRemaining missing values:")

    remaining_missing = df.isna().sum()

    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if remaining_missing.empty:
        print("No missing values remain.")
    else:
        print(remaining_missing)

    print("\nCleaned dataset shape:")
    print(df.shape)

    return df


def analyze_age_and_fare(df):

    print("\n" + "=" * 60)
    print("AGE AND FARE ANALYSIS")
    print("=" * 60)

    print("\nAge statistics:")
    print(df["age"].describe())

    print("\nFare statistics:")
    print(df["fare"].describe())

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]
    fare_skewness = df["fare"].skew()

    print("\nFare statistics:")
    print(f"Mean     : {fare_mean:.2f}")
    print(f"Median   : {fare_median:.2f}")
    print(f"Mode     : {fare_mode:.2f}")
    print(f"Skewness : {fare_skewness:.2f}")

    # -----------------------------
    # Age histogram
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["age"],
        bins=20
    )

    plt.title(
        "Distribution of Passenger Age"
    )

    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")

    plt.tight_layout()

    plt.savefig(
        "analytics/age_histogram.png"
    )

    plt.show()

    # -----------------------------
    # Fare histogram
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["fare"],
        bins=30
    )

    plt.title(
        "Distribution of Passenger Fare"
    )

    plt.xlabel("Fare")
    plt.ylabel("Number of Passengers")

    plt.tight_layout()

    plt.savefig(
        "analytics/fare_histogram.png"
    )

    plt.show()

    # -----------------------------
    # Age boxplot
    # -----------------------------

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        df["age"]
    )

    plt.title(
        "Age Boxplot"
    )

    plt.ylabel("Age")

    plt.tight_layout()

    plt.savefig(
        "analytics/age_boxplot.png"
    )

    plt.show()

    # -----------------------------
    # Fare boxplot
    # -----------------------------

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        df["fare"]
    )

    plt.title(
        "Fare Boxplot"
    )

    plt.ylabel("Fare")

    plt.tight_layout()

    plt.savefig(
        "analytics/fare_boxplot.png"
    )

    plt.show()

    # -----------------------------
    # IQR outlier function
    # -----------------------------

    def find_iqr_outliers(series):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound)
            |
            (series > upper_bound)
        ]

        return (
            q1,
            q3,
            iqr,
            lower_bound,
            upper_bound,
            len(outliers)
        )

    # -----------------------------
    # Age IQR
    # -----------------------------

    age_result = find_iqr_outliers(
        df["age"]
    )

    print("\nAge IQR analysis:")
    print(f"Q1             : {age_result[0]:.2f}")
    print(f"Q3             : {age_result[1]:.2f}")
    print(f"IQR            : {age_result[2]:.2f}")
    print(f"Lower bound    : {age_result[3]:.2f}")
    print(f"Upper bound    : {age_result[4]:.2f}")
    print(f"Number outliers: {age_result[5]}")

    # -----------------------------
    # Fare IQR
    # -----------------------------

    fare_result = find_iqr_outliers(
        df["fare"]
    )

    print("\nFare IQR analysis:")
    print(f"Q1             : {fare_result[0]:.2f}")
    print(f"Q3             : {fare_result[1]:.2f}")
    print(f"IQR            : {fare_result[2]:.2f}")
    print(f"Lower bound    : {fare_result[3]:.2f}")
    print(f"Upper bound    : {fare_result[4]:.2f}")
    print(f"Number outliers: {fare_result[5]}")


def analyze_survival(df):

    print("\n" + "=" * 60)
    print("SURVIVAL ANALYSIS")
    print("=" * 60)

    # -----------------------------
    # Survival by sex
    # -----------------------------

    survival_by_sex = (
        df.groupby("sex")["survived"]
        .mean()
        .reset_index()
    )

    survival_by_sex["survival_rate"] = (
        survival_by_sex["survived"] * 100
    )

    print("\nSurvival rate by sex:")
    print(
        survival_by_sex[
            ["sex", "survival_rate"]
        ]
    )

    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=survival_by_sex,
        x="sex",
        y="survival_rate"
    )

    plt.title(
        "Survival Rate by Sex"
    )

    plt.xlabel("Sex")
    plt.ylabel("Survival Rate (%)")

    plt.tight_layout()

    plt.savefig(
        "analytics/survival_by_sex.png"
    )

    plt.show()

    # -----------------------------
    # Survival by passenger class
    # -----------------------------

    survival_by_class = (
        df.groupby("pclass")["survived"]
        .mean()
        .reset_index()
    )

    survival_by_class["survival_rate"] = (
        survival_by_class["survived"] * 100
    )

    print("\nSurvival rate by passenger class:")
    print(
        survival_by_class[
            ["pclass", "survival_rate"]
        ]
    )

    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=survival_by_class,
        x="pclass",
        y="survival_rate"
    )

    plt.title(
        "Survival Rate by Passenger Class"
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")

    plt.tight_layout()

    plt.savefig(
        "analytics/survival_by_class.png"
    )

    plt.show()

    # -----------------------------
    # Survival by sex + class
    # -----------------------------

    survival_by_sex_class = (
        df.groupby(
            ["sex", "pclass"]
        )["survived"]
        .mean()
        .reset_index()
    )

    survival_by_sex_class[
        "survival_rate"
    ] = (
        survival_by_sex_class["survived"]
        * 100
    )

    print(
        "\nSurvival rate by sex and "
        "passenger class:"
    )

    print(
        survival_by_sex_class[
            [
                "sex",
                "pclass",
                "survival_rate"
            ]
        ]
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=survival_by_sex_class,
        x="pclass",
        y="survival_rate",
        hue="sex"
    )

    plt.title(
        "Survival Rate by Sex and Passenger Class"
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")

    plt.tight_layout()

    plt.savefig(
        "analytics/survival_by_sex_class.png"
    )

    plt.show()


def analyze_correlation(df):

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # The project specifically requires these
    # six columns and excludes adult_male and alone.
    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation_df = df[
        correlation_columns
    ]

    correlation_matrix = correlation_df.corr()

    print("\n6 x 6 correlation matrix:")
    print(correlation_matrix.round(3))

    # -----------------------------
    # Heatmap
    # -----------------------------

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "analytics/correlation_heatmap.png"
    )

    plt.show()

    # -----------------------------
    # Find two strongest absolute
    # off-diagonal correlations
    # -----------------------------

    absolute_matrix = correlation_matrix.abs()

    # Remove diagonal
    for column in absolute_matrix.columns:
        absolute_matrix.loc[
            column,
            column
        ] = 0

    correlation_pairs = (
        absolute_matrix
        .unstack()
        .sort_values(
            ascending=False
        )
    )

    # Remove duplicate pairs
    selected_pairs = []

    for (column_a, column_b), value in correlation_pairs.items():

        pair = frozenset(
            [column_a, column_b]
        )

        if column_a != column_b and pair not in [
            frozenset([a, b])
            for a, b, _ in selected_pairs
        ]:

            original_value = correlation_matrix.loc[
                column_a,
                column_b
            ]

            selected_pairs.append(
                (
                    column_a,
                    column_b,
                    original_value
                )
            )

        if len(selected_pairs) == 2:
            break

    print(
        "\nTwo strongest absolute "
        "off-diagonal correlations:"
    )

    for column_a, column_b, value in selected_pairs:

        print(
            f"{column_a} vs {column_b}: "
            f"{value:.3f}"
        )


if __name__ == "__main__":

    # 1. Load Titanic and save immediately
    df = load_and_save_titanic()

    # 2. Analyze missing values
    analyze_missing_values(df)

    # 3. Clean data
    cleaned_df = clean_data()

    # Save cleaned dataset
    cleaned_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # 4. Age and fare analysis
    analyze_age_and_fare(
        cleaned_df
    )

    # 5. Survival analysis
    analyze_survival(
        cleaned_df
    )

    # 6. Correlation analysis
    analyze_correlation(
        cleaned_df
    )

    print(
        f"\nCleaned dataset saved to {OUTPUT_FILE}"
    )