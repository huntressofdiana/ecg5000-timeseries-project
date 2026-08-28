import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

# Current file is probably inside:
# projects/ecg5000/src/
#
# parent       -> ecg5000
# parent.parent would go too far up

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "dataProcessed"

INPUT_PATH = OUTPUT_DIR / "ECG5000_CV_4000.txt"

TRAIN_OUTPUT = OUTPUT_DIR / "ECG5000_CV_TRAIN_3000.txt"

TEST_OUTPUT = OUTPUT_DIR / "ECG5000_CV_TEST_1000.txt"


# Make sure output directory exists
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD THE 4000 ECG SAMPLES
# ============================================================

data = np.loadtxt(INPUT_PATH)

print(
    "Original shape:",
    data.shape
)

# Expected:
# (4000, 141)
#
# 4000 ECG samples
# 1 label + 140 ECG time steps


# ============================================================
# 3. GET LABELS
# ============================================================

# First column contains:
# 1, 2, 3, 4, 5
#
# We only need this here so sklearn knows
# how to preserve the class proportions.

labels = data[:, 0]


# ============================================================
# 4. STRATIFIED 3000 / 1000 SPLIT
# ============================================================

train_data, test_data = train_test_split(

    data,

    train_size=3000,
    test_size=1000,

    # Preserve approximately the same
    # proportion of classes 1-5
    stratify=labels,

    # Same split every time we run the script
    random_state=42,

    shuffle=True,
)


# ============================================================
# 5. SAVE FILES
# ============================================================

np.savetxt(
    TRAIN_OUTPUT,
    train_data,
    fmt="%.18e",
)

np.savetxt(
    TEST_OUTPUT,
    test_data,
    fmt="%.18e",
)


# ============================================================
# 6. CHECK RESULTS
# ============================================================

print("\nTrain shape:")
print(train_data.shape)

print("\nTest shape:")
print(test_data.shape)


# ============================================================
# 7. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nTraining class counts:")

unique, counts = np.unique(
    train_data[:, 0],
    return_counts=True,
)

for label, count in zip(
    unique,
    counts
):
    print(
        f"Class {int(label)}: {count}"
    )


print("\nTest class counts:")

unique, counts = np.unique(
    test_data[:, 0],
    return_counts=True,
)

for label, count in zip(
    unique,
    counts
):
    print(
        f"Class {int(label)}: {count}"
    )


# ============================================================
# 8. PRINT WHERE FILES WERE SAVED
# ============================================================

print("\nFiles saved successfully:")

print(
    "Training:",
    TRAIN_OUTPUT
)

print(
    "Test:",
    TEST_OUTPUT
)