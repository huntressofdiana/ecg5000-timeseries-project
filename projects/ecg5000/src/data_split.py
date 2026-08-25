"""
Re-split the ECG5000 dataset.
 
The official ECG5000_TRAIN.txt / ECG5000_TEST.txt split gives only 500
training examples, which barely contains any of the rare classes
(class 5 especially). This script pools ALL 5000 examples together
and re-splits them into:
 
    1000 train
    1000 validation
    3000 test
 
using STRATIFIED splitting — meaning each split preserves the same
per-class proportions as the full 5000-example dataset. This gives
the new training set roughly double the rare-class examples of the
original 500-example split, without artificially inventing extra
data (that would be oversampling, a different technique).
 
Run this ONCE to produce three new .txt files. Your main training
script can then point at the new train/val files instead of the
originals.
"""
 
import numpy as np
from sklearn.model_selection import train_test_split
 
 
# --------------------------------------------------------
# 0. Config — adjust paths if needed
# --------------------------------------------------------
 
TRAIN_PATH = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\data\ECG5000_TRAIN.txt'
TEST_PATH = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\data\ECG5000_TEST.txt'
 
OUT_TRAIN_PATH = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\dataProcessed\ECG5000_TRAIN_1000.txt'
OUT_VAL_PATH = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\dataProcessed\ECG5000_VAL_1000.txt'
OUT_TEST_PATH = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\dataProcessed\ECG5000_TEST_3000.txt'
 
N_TRAIN = 1000
N_VAL = 1000
N_TEST = 3000  # should add up to the total example count (5000)
 
RANDOM_SEED = 42  # fixed seed = reproducible split every time you run this
 
 
def main():
 
    # --------------------------------------------------------
    # 1. Load and combine both original files
    # --------------------------------------------------------
    # We're discarding the original train/test boundary entirely and
    # re-partitioning from the full pool of 5000 examples.
 
    train_data = np.loadtxt(TRAIN_PATH)
    test_data = np.loadtxt(TEST_PATH)
 
    all_data = np.concatenate([train_data, test_data], axis=0)
 
    print(f"Total combined examples: {all_data.shape[0]}")
    assert all_data.shape[0] == N_TRAIN + N_VAL + N_TEST, (
        "N_TRAIN + N_VAL + N_TEST must equal the total number of examples"
    )
 
    # Column 0 = label, columns 1: = the 140 signal values
    labels = all_data[:, 0]
 
    # --------------------------------------------------------
    # 2. Show original class distribution (before splitting)
    # --------------------------------------------------------
    # This tells us just how rare class 5 (and others) really are.
 
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution across all 5000 examples:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {int(label)}: {count} examples ({count / len(labels) * 100:.1f}%)")
 
    # --------------------------------------------------------
    # 3. Stratified split: pool -> (train) + (val + test)
    # --------------------------------------------------------
    # train_test_split's `stratify` argument ensures each resulting
    # split has roughly the same per-class proportions as the input.
    # We do this in two steps because train_test_split only splits
    # into two pieces at a time.
 
    train_data_new, remaining_data = train_test_split(
        all_data,
        train_size=N_TRAIN,
        stratify=labels,          # preserve class proportions
        random_state=RANDOM_SEED, # reproducibility
    )
 
    # Re-extract labels for the remaining pool, to stratify the next split
    remaining_labels = remaining_data[:, 0]
 
    val_data_new, test_data_new = train_test_split(
        remaining_data,
        train_size=N_VAL,
        stratify=remaining_labels,
        random_state=RANDOM_SEED,
    )
 
    print(f"\nNew split sizes: train={train_data_new.shape[0]}, "
          f"val={val_data_new.shape[0]}, test={test_data_new.shape[0]}")
 
    # --------------------------------------------------------
    # 4. Confirm class 5 representation improved in training set
    # --------------------------------------------------------
 
    def class_5_count(data):
        return int((data[:, 0] == 5).sum())
 
    print(f"\nClass 5 count — original 500-example train file: "
          f"{class_5_count(train_data)}")
    print(f"Class 5 count — new {N_TRAIN}-example train file: "
          f"{class_5_count(train_data_new)}")
 
    # --------------------------------------------------------
    # 5. Save the three new files, same format as the originals
    # --------------------------------------------------------
    # np.savetxt writes plain space-separated numbers, matching the
    # format np.loadtxt expects to read back in your main script.
 
    np.savetxt(OUT_TRAIN_PATH, train_data_new)
    np.savetxt(OUT_VAL_PATH, val_data_new)
    np.savetxt(OUT_TEST_PATH, test_data_new)
 
    print(f"\nSaved:\n  {OUT_TRAIN_PATH}\n  {OUT_VAL_PATH}\n  {OUT_TEST_PATH}")
 
 
if __name__ == "__main__":
    main()