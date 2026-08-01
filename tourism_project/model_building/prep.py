"""
prep.py – Data Cleaning and Train/Test Splitting
=================================================
Loads the raw tourism dataset, drops non-predictive columns,
performs a stratified train/test split, and saves the splits
as CSV files for the model training job.
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────
DATA_PATH   = "tourism_project/data/tourism.csv"
TARGET_COL  = "ProdTaken"
TEST_SIZE   = 0.2
RANDOM_STATE = 42


def main():
    """Load, clean, split, and save the dataset."""
    print("Starting data preparation...")

    # ── 1. Load Raw Data ──────────────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # ── 2. Drop Unnecessary Columns ───────────────────────────────────────
    # 'Unnamed: 0' is a residual index column from CSV export.
    # 'CustomerID' is a unique identifier with no predictive power.
    cols_to_drop = ["Unnamed: 0", "CustomerID"]
    existing_drops = [c for c in cols_to_drop if c in df.columns]

    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)
        print(f"Dropped columns: {existing_drops}")
    else:
        print("No unnecessary columns found to drop.")

    print(f"   Remaining columns: {df.shape[1]}")

    # ── 3. Verify Target Column ───────────────────────────────────────────
    if TARGET_COL not in df.columns:
        print(f"Target column '{TARGET_COL}' not found!")
        sys.exit(1)

    # ── 4. Separate Features and Target ───────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"Features: {X.shape[1]} columns | Target: '{TARGET_COL}'")
    print(f"Target distribution:\n{y.value_counts().to_string()}")

    # ── 5. Stratified Train/Test Split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # Preserves the ~19% / ~81% target ratio in both sets
    )

    print(f"\nTrain/Test split (test_size={TEST_SIZE}, stratified):")
    print(f"   Train set: {X_train.shape[0]:,} samples")
    print(f"   Test set : {X_test.shape[0]:,} samples")

    # ── 6. Save Splits ────────────────────────────────────────────────────
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv",  index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv",  index=False)

    print("\nSaved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

    # ── 7. Verification ──────────────────────────────────────────────────
    print("\n# ── Split Verification ──")
    print(f"   Training target ratio:\n{y_train.value_counts(normalize=True).to_string()}")
    print(f"\n   Testing target ratio:\n{y_test.value_counts(normalize=True).to_string()}")


if __name__ == "__main__":
    main()
