import subprocess

from utils.paths import DATASET_PATH

scenes = [f'scene{i:04}_00' for i in range(10)]

for scene in scenes:
    command = f"unzip -o -q {DATASET_PATH}/scans/{scene}/{scene}_2d-instance-filt.zip -d {DATASET_PATH}/scans/{scene}"
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction of labels for {scene} completed successfully.")
    else:
        print(f"Extraction of labels for {scene} failed with error {process.returncode}")

