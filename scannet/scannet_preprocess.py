import os
import subprocess
from time import sleep

from utils.paths import DATASET_PATH

base_path = os.path.join(DATASET_PATH, 'scannet_frames_25k')
scenes = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
scenes.sort()
#scenes = [f'scene{i:04}_00' for i in range(10)]

for scene in scenes:
    sleep(10)
    command = f"python3 scannet/scannet_preprocess_utils.py --scene_folder ${{DATA_ROOT}}/scans/{scene} --scaled_image --semantics"

    print(f'Extract data for {scene}', command)
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction for {scene} completed successfully.")
    else:
        print(f"Extraction for {scene} failed with error {process.returncode}")