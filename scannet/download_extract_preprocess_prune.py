import os
import subprocess
import time

from utils.paths import DATASET_PATH

# Download scenes of pretraining 25k

subsample_factor = 10
base_path = os.path.join(DATASET_PATH, 'scannet_frames_25k')
scenes = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
scenes.sort()

# Download firsts 10 scenes for testing
#scenes = [f'scene{i:04}_00' for i in range(10)]


for scene in scenes:
    if scene < 'scene0706_01':
        continue

    # Download
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

    # Extract
    command = f"python3 scannet/extractor.py --filename ${{DATA_ROOT}}/scans/{scene}/{scene}.sens --output_path ${{DATA_ROOT}}/scans/{scene} --export_color_images --export_poses --export_intrinsics"

    print(f'Extract data for {scene}', command)
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction for {scene} completed successfully.")
    else:
        print(f"Extraction for {scene} failed with error {process.returncode}")

    command = f"unzip -o -q ${{DATA_ROOT}}/scans/{scene}/{scene}_2d-label-filt.zip -d ${{DATA_ROOT}}/scans/{scene}"
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

    # Preprocess
    command = f"python3 scannet/scannet_preprocess_utils.py --scene_folder ${{DATA_ROOT}}/scans/{scene} --scaled_image --semantics"

    print(f'Extract data for {scene}', command)
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction for {scene} completed successfully.")
    else:
        print(f"Extraction for {scene} failed with error {process.returncode}")

    # Prune
    scene_path = os.path.join(DATASET_PATH, 'scans', scene)
    # Remove useless zip
    if os.path.exists(f'{scene_path}/{scene}.sens'):
        os.remove(f'{scene_path}/{scene}.sens')
    if os.path.exists(f'{scene_path}/{scene}_2d-label.zip'):
        os.remove(f'{scene_path}/{scene}_2d-label.zip')
    if os.path.exists(f'{scene_path}/{scene}_2d-label-filt.zip'):
        os.remove(f'{scene_path}/{scene}_2d-label-filt.zip')
    if os.path.exists(f'{scene_path}/{scene}_2d-instance.zip'):
        os.remove(f'{scene_path}/{scene}_2d-instance.zip')

    # Subsample images
    folder = 'color'
    images = sorted([int(image.split('.')[0]) for image in os.listdir(os.path.join(scene_path, folder))])
    keep_images = images[::subsample_factor]
    for folder in ['color', 'color_scaled', 'label_40', 'label_40_scaled', 'label-filt']:
        extension = os.listdir(os.path.join(scene_path, folder))[0].split('.')[1]
        images = sorted([int(image.split('.')[0]) for image in os.listdir(os.path.join(scene_path, folder))])
        #keep_images = images[::subsample_factor]
        for image in images:
            if image in keep_images:
                continue
            try:
                os.remove(os.path.join(scene_path, folder, f'{image}.{extension}'))
            except:
                pass
    print(f'Prune for {scene} completed!!')