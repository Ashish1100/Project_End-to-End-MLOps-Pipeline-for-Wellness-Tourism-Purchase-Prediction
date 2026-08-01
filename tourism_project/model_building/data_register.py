"""
data_register.py – Dataset Validation & Registration
=====================================================
Reads the tourism.csv file from the repository data folder, validates
that all expected columns are present, and prints a concise summary.
This script acts as the first gate in the ML pipeline, ensuring data
integrity before any processing begins.
"""

import sys
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────
DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]


def main():
    """Run data validation and print dataset summary."""
    print("Starting dataset validation and registration...")

    # ── Load Dataset ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset loaded successfully from '{DATA_PATH}'")
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at '{DATA_PATH}'")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    # ── Column Validation ─────────────────────────────────────────────────
    actual_cols = set(df.columns)
    expected_set = set(EXPECTED_COLUMNS)
    missing = expected_set - actual_cols

    if missing:
        print(f"MISSING columns: {missing}")
        sys.exit(1)
    else:
        print(f"All {len(EXPECTED_COLUMNS)} expected columns are present.")

    extra = actual_cols - expected_set - {"Unnamed: 0"}
    if extra:
        print(f"Extra columns (will be handled later): {extra}")

    # ── Dataset Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Target  : ProdTaken")
    print(f"    - Not Purchased (0): {(df['ProdTaken']==0).sum():,}")
    print(f"    - Purchased     (1): {(df['ProdTaken']==1).sum():,}")

    print(f"\n  Missing Values per Column:")
    missing_vals = df.isnull().sum()
    has_missing = missing_vals[missing_vals > 0]
    if len(has_missing) == 0:
        print(f"    None detected ")
    else:
        for col in has_missing.index:
            print(f"    {col}: {has_missing[col]}")

    print(f"\n  Data Types:")
    for col in df.columns:
        print(f"    {col}: {df[col].dtype}")

    print(f"{'='*60}")
    print("Dataset registration and validation complete.")


if __name__ == "__main__":
    main()
