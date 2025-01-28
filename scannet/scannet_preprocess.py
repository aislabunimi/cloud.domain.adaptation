import subprocess

scenes = [f'scene{i:04}_00' for i in range(10)]

for scene in scenes:
    command = f"python3 scannet/scannet_preprocess_utils.py --scene_folder ${{DATA_ROOT}}/scans/{scene} --scaled_image --semantics"

    print(f'Extract data for {scene}', command)
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction for {scene} completed successfully.")
    else:
        print(f"Extraction for {scene} failed with error {process.returncode}")