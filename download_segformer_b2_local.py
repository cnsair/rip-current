# download_segformer_b2_local.py

# import os, certifi
# os.environ['SSL_CERT_FILE'] = certifi.where()
# from huggingface_hub import snapshot_download
# snapshot_download('nvidia/segformer-b2-finetuned-ade-512-512', local_dir='./segformer-b2-local')

import os, certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
from huggingface_hub import snapshot_download

snapshot_download(
    'nvidia/segformer-b2-finetuned-ade-512-512',
    local_dir='./segformer-b2-local',
    endpoint='https://hf-mirror.com'   # Use mirror
)