import shutil
import subprocess
import time

scenes = [f'scene{i:04}_00' for i in range(10)]

for scene in scenes:
    command = f"python3 scannet/extractor.py --filename ${{DATA_ROOT}}/scans/{scene}/{scene}.sens --output_path ${{DATA_ROOT}}/scans/{scene} --export_depth_images --export_color_images --export_poses --export_intrinsics"

    print(f'Extract data for {scene}', command)
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction for {scene} completed successfully.")
    else:
        print(f"Extraction for {scene} failed with error {process.returncode}")

    command = f"unzip -q ${{DATA_ROOT}}/scans/{scene}/{scene}_2d-label-filt.zip -d ${{DATA_ROOT}}/scans/{scene}"
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction of labels for {scene} completed successfully.")
    else:
        print(f"Extraction of labels for {scene} failed with error {process.returncode}")

    command = f"cp ${{DATA_ROOT}}/scannetv2-labels.combined.tsv ${{DATA_ROOT}}/scans/{scene}/scannetv2-labels.combined.tsv"
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Copy of labels_combined for {scene} completed successfully.")
    else:
        print(f"Copy of labels_combined for {scene} failed with error {process.returncode}")
