import subprocess
import time

scenes = [f'scene{i:04}_00' for i in range(10)]

for scene in scenes:
    command = (f"python3 official_download_script.py -o /media/antonazzi/Elements/scannet --id {scene}"
               f"")
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
    time.sleep(3)
    process.stdin.write(b'\n')
    process.stdin.flush()

    time.sleep(3)
    process.stdin.write(b'\n')
    process.stdin.flush()

    process.wait()


    if process.returncode == 0:
        print(f"Download for {scenes} completed successfully.")
    else:
        print(f"Failed to download {scene}. Return code: {process.returncode}")