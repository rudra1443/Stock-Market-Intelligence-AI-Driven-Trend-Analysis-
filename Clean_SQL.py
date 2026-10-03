import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# STOCK MARKET ANALYTICS - SQL READY DATA CLEANING
# ============================================================

# Original CSV location
SOURCE_FOLDER = Path("All_Org_data")

# SQL-ready output location
OUTPUT_FOLDER = Path("data") / "SQL_Ready"

# Create output folder if it does not exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STOCK MARKET ANALYTICS - SQL DATA PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# Check source folder
# ------------------------------------------------------------

if not SOURCE_FOLDER.exists():
    print("\nERROR: All_Org_data folder was not found.")
    print(f"Expected location: {SOURCE_FOLDER.resolve()}")
    raise SystemExit

# ------------------------------------------------------------
# Find CSV files
# ------------------------------------------------------------

csv_files = list(SOURCE_FOLDER.glob("*.csv"))

print(f"\nSource folder: {SOURCE_FOLDER.resolve()}")
print(f"CSV files found: {len(csv_files)}")

if len(csv_files) == 0:
    print("\nERROR: No CSV files found inside All_Org_data.")
    raise SystemExit

# ------------------------------------------------------------
# Process each CSV
# ------------------------------------------------------------

total_inf_replaced = 0

for file in csv_files:

    print("\n" + "-" * 70)
    print(f"Processing: {file.name}")

    try:

        # Read CSV
        df = pd.read_csv(file)

        original_rows = len(df)
        original_columns = len(df.columns)

        # ----------------------------------------------------
        # Remove spaces from column names
        # ----------------------------------------------------

        df.columns = df.columns.str.strip()

        # ----------------------------------------------------
        # Count Infinity values before replacing
        # ----------------------------------------------------

        inf_count = 0

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) > 0:
            inf_count = np.isinf(
                df[numeric_columns].to_numpy()
            ).sum()

        # ----------------------------------------------------
        # Replace Infinity values with NULL-compatible blanks
        # ----------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Also handle text versions of Infinity
        df = df.replace(
            [
                "inf",
                "-inf",
                "Infinity",
                "-Infinity"
            ],
            np.nan
        )

        # ----------------------------------------------------
        # Clean text columns
        # ----------------------------------------------------

        text_columns = df.select_dtypes(
            include="object"
        ).columns

        for column in text_columns:
            df[column] = df[column].apply(
                lambda value:
                value.strip()
                if isinstance(value, str)
                else value
            )

        # ----------------------------------------------------
        # Save SQL-ready file
        # ----------------------------------------------------

        output_file = OUTPUT_FOLDER / file.name

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        total_inf_replaced += int(inf_count)

        print(f"Rows       : {original_rows}")
        print(f"Columns    : {original_columns}")
        print(f"INF values : {inf_count}")
        print(f"Saved      : {output_file}")

    except Exception as error:

        print(f"\nERROR while processing {file.name}")
        print(error)

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SQL DATA PREPARATION COMPLETED")
print("=" * 70)

print(f"Files processed       : {len(csv_files)}")
print(f"INF values replaced   : {total_inf_replaced}")
print(f"SQL Ready folder      : {OUTPUT_FOLDER.resolve()}")

print("=" * 70)