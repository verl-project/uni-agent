mkdir wiki24-raw

hf download \
    --repo-type dataset \
    Upstash/wikipedia-2024-06-bge-m3 \
    --include 'data/en/*' \
    --local-dir wiki24-raw