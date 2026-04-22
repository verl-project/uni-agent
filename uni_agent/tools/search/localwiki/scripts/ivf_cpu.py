import numpy as np
import orjson as json
import faiss
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
import pyarrow.parquet as pq
import sys
from typing import List, Tuple, Optional

LOCAL_DATA_DIR = "/mnt/hdfs/went/wiki24-raw/data/en"
VECTOR_DIMENSION = 1024
INDEX_PATH = "/mnt/hdfs/went/wiki24/wiki24_faiss.index"
TEXT_DATA_PATH = "/mnt/hdfs/went/wiki24/wiki24_data.jsonl"

NLIST = 4096                # Number of inverted lists (cluster centroids)
TRAINING_SAMPLES = 2000000  # Vectors used to train the centroids
FAISS_METRIC = faiss.METRIC_L2

NUM_PROCESSES = mp.cpu_count()
if NUM_PROCESSES > 64:
    NUM_PROCESSES = 96


def process_parquet_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[List[bytes]]]:
    """Read a single Parquet file in a subprocess and extract embeddings and doc metadata."""
    try:
        table = pq.read_table(file_path)
        data_df = table.to_pandas()
        
        embeddings = np.stack(data_df["embedding"].to_numpy())
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        text_batch_lines = []
        for _, row in data_df.iterrows():
            doc_data = {
                "id": str(row["id"]),
                "url": row["url"],
                "title": row["title"],
                "text": row["text"],
            }
            json_bytes = json.dumps(doc_data) + b'\n'
            text_batch_lines.append(json_bytes)
            
        return embeddings, text_batch_lines

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        return None, None


def build_faiss_index_ivf_parallel():
    parquet_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*.parquet")))
    if not parquet_files:
        print(f"ERROR: no Parquet files found under '{LOCAL_DATA_DIR}'. Check the path.")
        return

    print(f"Found {len(parquet_files)} Parquet files to process.")

    # Stage 1: parallel I/O and preprocessing
    all_results = []
    vectors_processed = 0
    try:
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            print(f"Stage 1: Starting {NUM_PROCESSES} workers for Parallel Data Collection. Collecting to RAM...")

            results_iterator = pool.imap_unordered(process_parquet_file, parquet_files)

            pbar = tqdm(
                results_iterator, 
                total=len(parquet_files), 
                desc="Collecting All Data Chunks to RAM", 
                unit='file'
            )
            
            for embeddings, text_batch_lines in pbar:
                if embeddings is not None:
                    all_results.append((embeddings, text_batch_lines))
                    vectors_processed += len(embeddings)
                    pbar.set_postfix({"Total Docs": f"{vectors_processed:,}"})

            pbar.close()

        print(f"Stage 1 Complete. Total collected documents: {vectors_processed:,}. Now proceeding to training.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Aborting data collection stage.")
        return
    except Exception as e:
        print(f"\nStage 1 (Data Collection) failed: {e}")
        return
        
    if vectors_processed == 0:
        print("No vectors collected. Aborting index build.")
        return

    # Stage 2: index training (parallel K-Means)
    print(f"Stage 2: Training Index (NLIST={NLIST}). Using {TRAINING_SAMPLES:,} samples...")

    training_vectors_list = []
    current_count = 0
    for embeddings, _ in all_results:
        if current_count < TRAINING_SAMPLES:
            take = min(TRAINING_SAMPLES - current_count, len(embeddings))
            training_vectors_list.append(embeddings[:take])
            current_count += take

    training_matrix = np.concatenate(training_vectors_list, axis=0)

    quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
    final_index = faiss.IndexIVFFlat(quantizer, VECTOR_DIMENSION, NLIST, FAISS_METRIC)

    # Use all CPU cores during K-Means training, then restore to 1 for the add stage.
    faiss.omp_set_num_threads(NUM_PROCESSES)
    final_index.train(training_matrix)
    faiss.omp_set_num_threads(1)
    print("Index Training Complete.")

    # Stage 3: add vectors and write JSONL
    print("Stage 3: Adding Vectors and Writing JSONL.")
    current_idx = 0
    try:
        with open(TEXT_DATA_PATH, 'wb') as f_out:
            pbar = tqdm(all_results, desc='Adding to Index', unit='batch')
            for embeddings, text_batch_lines in pbar:
                f_out.writelines(text_batch_lines)
                final_index.add(embeddings)
                current_idx += len(embeddings)

                pbar.set_postfix({"Total Docs": f"{current_idx:,}"})

            pbar.close()

    except Exception as e:
        print(f"\nStage 3 (Add/Write) failed: {e}")
        print(f"Vectors processed so far: {final_index.ntotal}")

    if final_index.ntotal > 0:
        print(f"\nFinalizing and saving FAISS index to {INDEX_PATH}...")
        faiss.write_index(final_index, INDEX_PATH)
        print(f"Index successfully saved with {final_index.ntotal:,} vectors.")
    else:
        print("Index is empty, skipping save.")


if __name__ == "__main__":
    # 'spawn' is safer than 'fork' for FAISS + multiprocessing.
    if os.name != 'nt':
        mp.set_start_method('spawn', force=True)

    build_faiss_index_ivf_parallel()