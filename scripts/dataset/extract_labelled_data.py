import os
import shutil

from omegaconf import OmegaConf,open_dict
from scripts.paths import dataset_path
from scannetpp.semantic.prep.rasterize_single_scene import main as rasterize_scene
from scannetpp.semantic.prep.single_scene_semantics_2d import main as generate_semantic_2d

scene_file = os.path.join(dataset_path, 'splits', 'nvs_sem_train.txt')

conf_raster = OmegaConf.create(
    {
        # release data downloaded to disk
        'data_root': os.path.join(dataset_path, 'data'),
        'scene_list_file': scene_file,
        'filter_scenes': [],

        # image type - iphone or dslr
        'image_type': 'dslr',

        'no_log': True,

        # subsample images
        'subsample_factor': 100,
        # downsample images and rasterize
        'image_downsample_factor': 3,

        'rasterout_dir': os.path.join(dataset_path, 'raster'),

        'limit_images': None,

        'wandb_group': None,
        'wandb_notes': None,

        'batch_size': 3,
        'limit_batches': None,
        'skip_existing': False,
    }
)
OmegaConf.set_struct(conf_raster, True)

conf_semantic = OmegaConf.create(
    {
        ############### input data #################
        # release data downloaded to disk
        'data_root': os.path.join(dataset_path, 'data'),
        'scene_list_file': scene_file,
        'rasterout_dir': os.path.join(dataset_path, 'raster'),
        'visiblity_cache_dir': 'visibility_cache',
        'filter_scenes': [],
        'exclude_scenes': [],
        # image type - iphone or dslr
        'image_type': 'dslr',
        'undistort_dslr': True,
        'create_visiblity_cache_only': False,
        ############### hyperparams #################
        # use topk views with visiblity of object vertices
        'visibility_topk': 3,
        # min size of the bbox of an object (each side must be greater than this in pixels)
        'bbox_min_side_pix': 50,
        # subsample images
        'subsample_factor': 100,
        # atleast this fraction of the object's vertices should be visible in the image
        # set to 0 to ignore threshold
        'obj_visible_thresh': 0.1,
        # object should cover atleast this fraction of the image's pixels
        # set to 0 to ignore threshold
        'obj_pixel_thresh': 0.00,
        # object should be within this distance from the camera (meters) (set large number to include all objects)
        'obj_dist_thresh': 999,
        # expand the bbox by this fraction in each direction
        'bbox_expand_factor': 0.1,
        ############### output #################
        'save_dir_root': os.path.join(dataset_path, 'semantic'),
        'save_dir': 'semantics_2d',
        ############### dbg #################
        'dbg': {'viz_obj_ids': True, 'save_obj_ids': True}
    }
)
OmegaConf.set_struct(conf_semantic, True)

scenes = []
with open(scene_file) as file:
    scenes = [line.rstrip() for line in file]

for scene in scenes:
    if not os.path.exists(os.path.join(dataset_path, 'data', scene)):
        continue
    conf_raster.filter_scenes = [scene]
    conf_semantic.filter_scenes = [scene]
    rasterize_scene(conf_raster)

    generate_semantic_2d(conf_semantic)
    shutil.rmtree(os.path.join(dataset_path, 'raster', 'dslr', scene))
