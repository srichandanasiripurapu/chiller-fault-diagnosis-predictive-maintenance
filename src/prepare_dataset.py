
from pathlib import Path
from collections import Counter
import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "chiller_sample.parquet"
TEMP_FILE = OUTPUT_DIR / "chiller_sample.tmp.parquet"

TARGET_ROWS_PER_FILE = 1_000_000
BATCH_SIZE = 100_000
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Uniform random sampling using batches
# --------------------------------------------------

def reservoir_sample(file_path, target_rows, batch_size, rng):
    """
    Select exactly target_rows uniformly at random
    without loading the entire Parquet file into memory.
    """

    parquet_file = pq.ParquetFile(file_path)

    if parquet_file.metadata.num_rows < target_rows:
        raise ValueError(
            f"{file_path.name} contains fewer than "
            f"{target_rows:,} rows."
        )

    reservoir_parts = []
    reservoir = None
    collected = 0
    seen = 0

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size
    ):
        frame = record_batch.to_pandas()
        offset = 0

        # Fill the initial reservoir in batches.
        if reservoir is None:
            take = min(
                target_rows - collected,
                len(frame)
            )

            if take > 0:
                reservoir_parts.append(
                    frame.iloc[:take].copy()
                )

                collected += take
                seen += take
                offset = take

            # Build the reservoir only once it is full.
            if collected == target_rows:
                reservoir = pd.concat(
                    reservoir_parts,
                    ignore_index=True
                )
                reservoir_parts = []

        # Use vectorized reservoir sampling for the
        # remaining rows instead of updating one row
        # at a time.
        if reservoir is not None and offset < len(frame):
            remaining = frame.iloc[offset:]
            count = len(remaining)

            # For each row, select a random reservoir
            # position using Algorithm R.
            row_numbers = seen + np.arange(count)
            random_positions = rng.integers(
                low=0,
                high=row_numbers + 1
            )

            replace_mask = random_positions < target_rows

            if replace_mask.any():
                candidate_positions = np.flatnonzero(
                    replace_mask
                )
                slots = random_positions[replace_mask]

                # If several rows target the same slot
                # in this batch, keep the last one.
                order = np.argsort(slots, kind="stable")
                sorted_slots = slots[order]

                last_in_group = np.r_[
                    sorted_slots[1:] != sorted_slots[:-1],
                    True
                ]

                chosen_order = order[last_in_group]
                unique_slots = slots[chosen_order]

                chosen_rows = candidate_positions[chosen_order]
                candidates = remaining.iloc[chosen_rows]

                # Assign column by column to preserve
                # the original column data types.
                for column_number in range(len(reservoir.columns)):
                    reservoir.iloc[
                        unique_slots, column_number
                    ] = candidates.iloc[
                        :, column_number
                    ].to_numpy()

            seen += count

        del frame

    if reservoir is None or len(reservoir) != target_rows:
        raise RuntimeError(
            f"Sampling failed for {file_path.name}. "
            f"Expected {target_rows:,} rows."
        )

    return reservoir


# --------------------------------------------------
# 3. Find source files
# --------------------------------------------------

source_files = sorted(DATA_DIR.glob("*.parquet"))

if not source_files:
    raise FileNotFoundError(
        f"No source Parquet files found in {DATA_DIR}"
    )

print(f"Source files found: {len(source_files)}")
print(f"Rows to sample per file: {TARGET_ROWS_PER_FILE:,}")
print(
    "Expected output rows: "
    f"{len(source_files) * TARGET_ROWS_PER_FILE:,}"
)
print(f"Batch size: {BATCH_SIZE:,}")
print("\nOriginal source files will not be modified.\n")


# --------------------------------------------------
# 4. Create the temporary dataset
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)
writer = None

# Remove an incomplete temporary file from a previous run.
if TEMP_FILE.exists():
    TEMP_FILE.unlink()

try:
    for file_number, file_path in enumerate(source_files, start=1):
        print(
            f"[{file_number}/{len(source_files)}] "
            f"Sampling {file_path.name}..."
        )

        sample = reservoir_sample(
            file_path=file_path,
            target_rows=TARGET_ROWS_PER_FILE,
            batch_size=BATCH_SIZE,
            rng=rng
        )

        table = pa.Table.from_pandas(
            sample,
            preserve_index=False
        ).replace_schema_metadata(None)

        # Create the writer from the first sampled file.
        if writer is None:
            writer = pq.ParquetWriter(
                TEMP_FILE,
                table.schema,
                compression="snappy"
            )

        writer.write_table(
            table,
            row_group_size=BATCH_SIZE
        )

        print(f"    Selected {len(sample):,} rows")

        del sample, table
        gc.collect()

finally:
    if writer is not None:
        writer.close()


# --------------------------------------------------
# 5. Validate the completed temporary dataset
# --------------------------------------------------

print("\nValidating the temporary dataset...")

if not TEMP_FILE.exists():
    raise FileNotFoundError(
        "The temporary output file was not created."
    )

result_file = pq.ParquetFile(TEMP_FILE)
actual_rows = result_file.metadata.num_rows
expected_rows = len(source_files) * TARGET_ROWS_PER_FILE

if actual_rows != expected_rows:
    raise RuntimeError(
        f"Row-count validation failed: expected "
        f"{expected_rows:,}, found {actual_rows:,}. "
        "The existing output has been preserved."
    )

fault_counts = Counter()
severity_counts = Counter()
missing_labels = 0

for batch in result_file.iter_batches(
    columns=["fault", "severity_level"],
    batch_size=BATCH_SIZE
):
    labels = batch.to_pandas()

    missing_labels += int(
        labels[["fault", "severity_level"]]
        .isna()
        .any(axis=1)
        .sum()
    )

    fault_counts.update(
        labels["fault"].value_counts().to_dict()
    )

    severity_counts.update(
        labels["severity_level"].value_counts().to_dict()
    )

    del labels

if missing_labels:
    raise RuntimeError(
        f"Validation failed: {missing_labels:,} rows "
        "have missing fault or severity labels. "
        "The existing output has been preserved."
    )

print(f"Validated rows: {actual_rows:,}")
print(f"Missing target labels: {missing_labels}")

print("\nFault category counts:")
for label, count in sorted(fault_counts.items(), key=lambda x: str(x[0])):
    print(f"  {label}: {count:,}")

print("\nSeverity counts:")
for label, count in sorted(severity_counts.items(), key=lambda x: str(x[0])):
    print(f"  {label}: {count:,}")


# --------------------------------------------------
# 6. Replace the old sample only after validation
# --------------------------------------------------

TEMP_FILE.replace(OUTPUT_FILE)

print("\nDataset preparation completed successfully.")
print(f"Final row count: {actual_rows:,}")
print(f"Saved to: {OUTPUT_FILE}")
