import gdown
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SAVE_DIR = ROOT_DIR

os.makedirs(SAVE_DIR, exist_ok=True)
url = "https://drive.google.com/drive/u/0/folders/1x-WUcYYZUGJwJ8BOaJxVAk6k3D286Qmy"
gdown.download_folder(url, output=SAVE_DIR, quiet=False)