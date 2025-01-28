# cloud.domain.adaptation

This work is based on the paper *Unsupervised Continual Semantic Adaptation through Neural Rendering* ([official repo](https://github.com/ethz-asl/ucsa_neural_rendering)).
## Installation instruction
With python 3.10:
* Define the CUDA_HOME environment variable inside the .bashrc file `export CUDA_HOME="/usr/local/cuda-12.1"`
* Install the repo using pip `pip3 install -e .`

## Dataset preparation
At first request the official script to download Scannet [here](https://github.com/ScanNet/ScanNet/tree/master) and copy it inside a python file in `scannet/official_download_script.py`

* Select the directory in which you want to download the dataset with `export DATA_ROOT=~/scannet`
* Download the labels with `python3 scannet/official_download_script.py --label_map -o ${DATA_ROOT}`
* Download the dataset of the 25k frames of the scenes from 10 to 707 using the official script with `python3 scannet/official_download_script.py --preprocessed_frames -o ${DATA_ROOT}` 
* Extract the content in a subfolder called `${DATA_ROOT}/scannet_frames_25k` and copy the file `scannetv2-labels.combined.tsv` inside it
* Download the data of the scenes from 0 to 10 with `python3 scannet/download_scenes.py`
* Extract all the sensor data for each of the downloaded scenes from 0000 to 0009. To do this, run `python3 scannet/extract_data.py`
* At the end of the process, the `${DATA_ROOT}` folder should contain _at least_ the following data, structured as below:

    ```shell
    scannet
      scannet_frames_25k
        scene0010_00
          color
            000000.jpg
            ...
            XXXXXX.jpg
          label
            000000.png
            ...
            XXXXXX.png
        ...
        ...
        scene0706_00
          ...
        scannetv2-labels.combined.tsv
      scans
        scene0000_00
          color
            000000.jpg
            ...
            XXXXXX.jpg
          depth
            000000.png
            ...
            XXXXXX.png
          label-filt
            000000.png
            ...
            XXXXXX.png
          pose
            000000.txt
            ...
            XXXXXX.txt
          intrinsics
            intriniscs_color.txt
            intrinsics_depth.txt
          scannetv2-labels.combined.tsv
        ...
        scene0009_00
          ...
    ```
* Copy the files [`split.npz`](./scannet/split.npz) and [`split_cl.npz`](./scannet/split_cl.npz) to the `${DATA_ROOT}/scannet_frames_25k` folder. These files contain the indices of the samples that define the train/validation splits used in pre-training and to form the replay buffer in continual adaptation, to ensure reproducibility.

