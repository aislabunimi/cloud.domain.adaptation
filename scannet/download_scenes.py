import os
import subprocess
import time

from utils.paths import DATASET_PATH

# Download scenes of pretraining 25k

base_path = os.path.join(DATASET_PATH, 'scannet_frames_25k')
scenes = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
scenes.sort()

# Download firsts 10 scenes for testing
#scenes = [f'scene{i:04}_00' for i in range(10)]

for scene in scenes:
    command = ("python3 scannet/official_download_script.py -o ${DATA_ROOT} --id " + scene)
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
    time.sleep(1)
    process.stdin.write(b'\n')
    process.stdin.flush()

    time.sleep(1)
    process.stdin.write(b'\n')
    process.stdin.flush()

    process.wait()


    if process.returncode == 0:
        print(f"Download for {scene} completed successfully.")
    else:
        print(f"Failed to download {scene}. Return code: {process.returncode}")