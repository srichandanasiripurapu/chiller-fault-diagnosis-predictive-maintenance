
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

# --------------------------------------------------
# 1. Load dataset metadata
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_DIR / "data" / "processed" / "chiller_sample.parquet"
)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

parquet = pq.ParquetFile(DATASET_PATH)

print("=" * 60)
print("LABEL RELATIONSHIP AND DATA-LEAKAGE CHECK")
print("=" * 60)
print(f"Rows: {parquet.metadata.num_rows:,}")

# --------------------------------------------------
# 2. Count relationships between labels
# --------------------------------------------------

fault_counts = Counter()
fault_id_by_fault = {}
faults_by_id = {}
severity_by_fault = {}
path_labels_by_fault = {}
source_files_by_fault = {}

for batch in parquet.iter_batches(
    columns=[
        "fault",
        "fault_id",
        "severity_level",
        "path_label",
        "source_file",
    ],
    batch_size=100_000,
):
    frame = batch.to_pandas()

    for row in frame.itertuples(index=False):
        fault, fault_id, severity, path_label, source_file = row

        fault_counts[fault] += 1

        fault_id_by_fault.setdefault(fault, set()).add(fault_id)
        faults_by_id.setdefault(fault_id, set()).add(fault)

        severity_by_fault.setdefault(fault, Counter())[severity] += 1
        path_labels_by_fault.setdefault(fault, set()).add(path_label)
        source_files_by_fault.setdefault(fault, set()).add(source_file)

    del frame

# --------------------------------------------------
# 3. Print fault and ID relationships
# --------------------------------------------------

print("\nFault counts:")
for fault, count in sorted(fault_counts.items()):
    print(f"  {fault}: {count:,}")

print("\nFault-to-ID relationship:")
for fault, ids in sorted(fault_id_by_fault.items()):
    print(f"  {fault}: fault_id values = {sorted(ids)}")

print("\nID-to-fault relationship:")
for fault_id, faults in sorted(faults_by_id.items()):
    print(f"  fault_id {fault_id}: faults = {sorted(faults)}")

# --------------------------------------------------
# 4. Print severity distributions by fault
# --------------------------------------------------

print("\nSeverity counts within each fault:")
for fault, counts in sorted(severity_by_fault.items()):
    print(f"  {fault}: {dict(sorted(counts.items()))}")

# --------------------------------------------------
# 5. Check whether file/path labels identify faults
# --------------------------------------------------

print("\nSource files associated with each fault:")
for fault, files in sorted(source_files_by_fault.items()):
    print(f"  {fault}: {sorted(files)}")

print("\nPath labels associated with each fault:")
for fault, labels in sorted(path_labels_by_fault.items()):
    print(f"  {fault}: {sorted(labels)}")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)
print("This script only reads data; it does not modify the dataset.")
