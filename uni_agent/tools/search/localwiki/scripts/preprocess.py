"""Convert the JSONL corpus into pickled `corpus.pkl` and `url_to_ids.pkl` for the retrieval server."""
import pickle
import orjson as json
import os


def preprocess_corpus(jsonl_path, output_dir):
    docs = []
    url_to_ids = {}

    with open(jsonl_path, 'rb') as f:
        for idx, line in enumerate(f):
            doc = json.loads(line)
            docs.append(doc)
            url = doc.get("url")
            if url:
                url_to_ids.setdefault(url, []).append(idx)

    with open(f"{output_dir}/corpus.pkl", 'wb') as f:
        pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{output_dir}/url_to_ids.pkl", 'wb') as f:
        pickle.dump(url_to_ids, f, protocol=pickle.HIGHEST_PROTOCOL)


wiki_path_prefix = os.getenv("DATA_ROOT", "/mnt/hdfs/went") + "/wiki24"
preprocess_corpus(f"{wiki_path_prefix}/wiki24_data.jsonl", f"{wiki_path_prefix}/wiki24_preprocessed")