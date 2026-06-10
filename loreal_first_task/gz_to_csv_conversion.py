import glob
import gzip
import os
import polars as pl
from pathlib import Path
from datetime import datetime


# ── HARDCODED OUTPUT PREFIX ──
OUTPUT_PREFIX = "Loreal-USA-OAP-BaseFile_Daily_"


def generate_output_filename(output_dir: str) -> str:
    """
    Generate output file name with current timestamp.
    """
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{OUTPUT_PREFIX}{current_timestamp}.csv"
    return os.path.join(output_dir, output_filename)


def merge_gz_to_csv(
    input_dir: str,
    output_dir: str,
    separator: str = "|",
    gz_pattern: str = "*.gz",
) -> None:
    """
    Decompress n .gz files (pipe-separated), merge them into one
    comma-separated .csv file using Polars.

    Args:
        input_dir   : Directory containing .gz files.
        output_dir  : Directory where the final merged .csv will be written.
        separator   : Delimiter used inside the .gz files (default '|').
        gz_pattern  : Glob pattern to match .gz files.
    """

    gz_files = sorted(glob.glob(os.path.join(input_dir, gz_pattern)))

    if not gz_files:
        print(f"No .gz files found in '{input_dir}' matching '{gz_pattern}'.")
        return

    print(f"Found {len(gz_files)} .gz file(s) in '{input_dir}'.\n")

    # ── Generate output file name with current timestamp ──
    output_file = generate_output_filename(output_dir)
    print(f"Output file will be: {output_file}\n")

    frames: list[pl.DataFrame] = []

    for idx, gz_path in enumerate(gz_files, start=1):
        file_name = os.path.basename(gz_path)
        print(f"[{idx}/{len(gz_files)}] Processing: {file_name} ...")

        try:
            # ---------------------------------------------------
            # 1. Decompress .gz → raw bytes (in-memory)
            # ---------------------------------------------------
            with gzip.open(gz_path, "rb") as f:
                raw_bytes = f.read()

            # ---------------------------------------------------
            # 2. Read pipe-separated content with Polars
            # ---------------------------------------------------
            df = pl.read_csv(
                source=raw_bytes,
                separator=separator,
                has_header=True,
                infer_schema_length=0,        # all columns as String
                truncate_ragged_lines=True,
                ignore_errors=True,
            )

            print(f"   → Rows: {df.height:,}  |  Columns: {df.width}")
            frames.append(df)

        except Exception as e:
            print(f"   ⚠ SKIPPED (error): {e}")
            continue

    if not frames:
        print("\nNo data frames to merge. Exiting.")
        return

    # ---------------------------------------------------
    # 3. Merge all frames vertically
    # ---------------------------------------------------
    print("\nMerging all frames ...")
    merged_df = pl.concat(frames, how="diagonal_relaxed")
    print(f"Merged shape  → Rows: {merged_df.height:,}  |  Columns: {merged_df.width}")

    # ---------------------------------------------------
    # 4. Write final comma-separated .csv
    # ---------------------------------------------------
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    merged_df.write_csv(output_file, separator=",")
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print(f"\n✅ Output written to : {output_file}")
    print(f"   File size         : {file_size_mb:.2f} MB")
    print(f"   Total rows        : {merged_df.height:,}")
    print(f"   Total columns     : {merged_df.width}")


# ─────────────────────── ENTRY POINT ───────────────────────
if __name__ == "__main__":

    # ── CONFIGURE THESE ──────────────────────────────────────
    INPUT_DIR  = "./input_files_folder/loreal_OAP_files"     # folder with your .gz files
    OUTPUT_DIR = "./output_files_folder/"       # folder for the merged CSV
    # ─────────────────────────────────────────────────────────

    merge_gz_to_csv(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        separator="|",
    )