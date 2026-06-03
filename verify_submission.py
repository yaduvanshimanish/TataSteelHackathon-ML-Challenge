"""
verify_submission.py — Validate expected_submission.csv before uploading.
Usage: python verify_submission.py --submission expected_submission.csv --test test.csv
"""
import argparse
import pandas as pd
import sys

def verify(sub_path, test_path):
    print(f"Verifying: {sub_path}")

    sub  = pd.read_csv(sub_path)
    test = pd.read_csv(test_path)

    errors = []

    # Shape
    if sub.shape != (339, 2):
        errors.append(f"Shape: expected (339, 2), got {sub.shape}")

    # Columns
    if list(sub.columns) != ['CoilID', 'Y']:
        errors.append(f"Columns: expected ['CoilID','Y'], got {list(sub.columns)}")

    # CoilIDs
    test_ids = set(test['CoilID'].tolist())
    sub_ids  = set(sub['CoilID'].tolist())
    if test_ids != sub_ids:
        missing = test_ids - sub_ids
        extra   = sub_ids - test_ids
        if missing: errors.append(f"Missing CoilIDs: {missing}")
        if extra:   errors.append(f"Extra CoilIDs: {extra}")

    # Y values
    invalid_y = sub[~sub['Y'].isin([0, 1])]
    if len(invalid_y):
        errors.append(f"Invalid Y values: {invalid_y['Y'].unique().tolist()}")

    # Distribution
    n1 = (sub['Y'] == 1).sum()
    n0 = (sub['Y'] == 0).sum()
    fpr_max = n1 - 265  # assuming 265 TPs (best case)
    fpr_pct = fpr_max / 74 * 100

    print(f"\n  Defects (Y=1)    : {n1}")
    print(f"  Non-defects (Y=0): {n0}")
    print(f"  Estimated FPR    : <={fpr_pct:.1f}% (need < 10%)")

    if fpr_pct >= 10.0:
        errors.append(f"WARNING: If all non-TP predictions are FP, FPR={fpr_pct:.1f}% >= 10%")

    if errors:
        print("\n[ERROR] ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Submission file is valid. Ready to upload.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='expected_submission.csv')
    parser.add_argument('--test',       default='test.csv')
    args = parser.parse_args()
    verify(args.submission, args.test)
