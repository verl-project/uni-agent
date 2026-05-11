"""Convert the JSONL corpus into pickled `corpus.pkl` and `url_to_ids.pkl` for the retrieval server.

Defaults read JSONL from ``$DATA_ROOT/wiki24/wiki24_data.jsonl`` and write the
pickled corpus to ``$DATA_ROOT/wiki24/wiki24_preprocessed/``. Override either
explicitly via CLI args or via the DATA_ROOT / TEXT_DATA_PATH / CORPUS_DIR env
vars.
"""

import argparse
import os
import pickle

import orjson as json


def preprocess_corpus(jsonl_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    docs = []
    url_to_ids: dict[str, list[int]] = {}

    with open(jsonl_path, "rb") as f:
        for idx, line in enumerate(f):
            doc = json.loads(line)
            docs.append(doc)
            url = doc.get("url")
            if url:
                url_to_ids.setdefault(url, []).append(idx)

    with open(os.path.join(output_dir, "corpus.pkl"), "wb") as f:
        pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "url_to_ids.pkl"), "wb") as f:
        pickle.dump(url_to_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Preprocessed {len(docs):,} docs ({len(url_to_ids):,} unique urls) into {output_dir}")


if __name__ == "__main__":
    data_root = os.environ.get("DATA_ROOT", os.path.expanduser("~/uni_agent_data"))
    parser = argparse.ArgumentParser(description="Preprocess wiki JSONL into pickled corpus.")
    parser.add_argument(
        "--jsonl",
        default=os.environ.get("TEXT_DATA_PATH", os.path.join(data_root, "wiki24", "wiki24_data.jsonl")),
        help="Path to the source corpus JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("CORPUS_DIR", os.path.join(data_root, "wiki24", "wiki24_preprocessed")),
        help="Directory to write corpus.pkl and url_to_ids.pkl into.",
    )
    args = parser.parse_args()

    preprocess_corpus(args.jsonl, args.out_dir)
