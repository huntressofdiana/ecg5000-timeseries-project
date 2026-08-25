from pathlib import Path
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)


# ============================================================
# PATHS
# ============================================================

# data_split.py is inside:
# ecg5000/src/data_split.py

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = PROJECT_DIR / "dataProcessed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


train_path = DATA_DIR / "ECG5000_TRAIN.txt"
test_path = DATA_DIR / "ECG5000_TEST.txt"


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("Loading data from:")
print(train_path)
print(test_path)


train_data = np.loadtxt(train_path)
test_data = np.loadtxt(test_path)


print("\nOriginal shapes:")
print("Train:", train_data.shape)
print("Test:", test_data.shape)


# ============================================================
# COMBINE ALL 5000 SAMPLES
# ============================================================

all_data = np.vstack([
    train_data,
    test_data
])


print("\nCombined shape:")
print(all_data.shape)


# First column = class label
labels = all_data[:, 0]


# ============================================================
# 4000 CROSS-VALIDATION / 1000 FINAL TEST
# ============================================================

cv_data, final_test = train_test_split(
    all_data,
    test_size=1000,
    random_state=42,
    shuffle=True,
    stratify=labels,
)


print("\nNew split:")
print("Cross-validation:", cv_data.shape)
print("Final test:", final_test.shape)


# Save complete CV dataset
np.savetxt(
    OUTPUT_DIR / "ECG5000_CV_4000.txt",
    cv_data
)

# Save untouched final test set
np.savetxt(
    OUTPUT_DIR / "ECG5000_FINAL_TEST_1000.txt",
    final_test
)


# ============================================================
# CREATE 5 STRATIFIED FOLDS
# ============================================================

cv_labels = cv_data[:, 0]

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


for fold_number, (train_idx, val_idx) in enumerate(
    skf.split(cv_data, cv_labels),
    start=1,
):

    fold_train = cv_data[train_idx]
    fold_val = cv_data[val_idx]

    print(
        f"\nFold {fold_number}: "
        f"train={len(fold_train)}, "
        f"validation={len(fold_val)}"
    )

    # Save validation fold by itself
    np.savetxt(
        OUTPUT_DIR
        / f"ECG5000_FOLD_{fold_number}_800.txt",
        fold_val
    )

    # Save training portion for this fold
    np.savetxt(
        OUTPUT_DIR
        / f"ECG5000_FOLD_{fold_number}_TRAIN_3200.txt",
        fold_train
    )

    # Save validation portion for this fold
    np.savetxt(
        OUTPUT_DIR
        / f"ECG5000_FOLD_{fold_number}_VAL_800.txt",
        fold_val
    )


# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("Finished!")
print("Files saved to:")
print(OUTPUT_DIR)
print("======================================")