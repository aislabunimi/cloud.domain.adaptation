'''
Get 3D semantics onto 2D images using precomputed rasterization
'''

from common.utils.image import get_img_crop, load_image, save_img, viz_ids, save_ids
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
from omegaconf import DictConfig
import wandb

from tqdm import tqdm
import open3d as o3d
import torch
import numpy as np
import cv2

from common.utils.dslr import compute_undistort_intrinsic
from common.utils.colmap import get_camera_images_poses, camera_to_intrinsic
from common.utils.anno import get_bboxes_2d, get_visiblity_from_cache, get_vtx_prop_on_2d, load_anno_wrapper
from common.file_io import read_txt_list
from common.scene_release import ScannetppScene_Release


@hydra.main(version_base=None, config_path="../configs", config_name='semantics_2d')
def main(cfg : DictConfig) -> None:
    #print('Config:', cfg)

    # get scene list
    scene_list = read_txt_list(cfg.scene_list_file)
    #print('Scenes in list:', len(scene_list))

    if cfg.get('filter_scenes'):
        scene_list = [s for s in scene_list if s in cfg.filter_scenes]
        #print('Filtered scenes:', len(scene_list))
    if cfg.get('exclude_scenes'):
        scene_list = [s for s in scene_list if s not in cfg.exclude_scenes]
        print('After excluding scenes:', len(scene_list))

    # root + runname + savedir
    save_dir = Path(cfg.save_dir_root) / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    #print('Save to dir:', save_dir)

    #img_crop_dir =  save_dir / 'img_crops'
    #bbox_img_dir =  save_dir / 'img_bbox'


    rasterout_dir = Path(cfg.rasterout_dir) / cfg.image_type

    # go through scenes
    for scene_id in tqdm(scene_list, desc='scene'):
        scene_dir = save_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        viz_obj_ids_dir = scene_dir / 'viz_obj_ids'
        viz_obj_ids_dir.mkdir(parents=True, exist_ok=True)

        obj_ids_dir = scene_dir / 'obj_ids'
        obj_ids_dir.mkdir(parents=True, exist_ok=True)

        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
        # get object ids on the mesh vertices
        anno = load_anno_wrapper(scene)

        # create visibility cache to pick topk images where an object is visible
        #visibility_data = get_visiblity_from_cache(scene, rasterout_dir, cfg.visiblity_cache_dir, cfg.image_type, cfg.subsample_factor, cfg.undistort_dslr, anno=anno)
        if cfg.create_visiblity_cache_only:
            print(f'Created visibility cache for {scene_id}')
            continue

        vtx_obj_ids = anno['vertex_obj_ids']
        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 

        obj_ids = np.unique(vtx_obj_ids)
        # remove 0
        obj_ids = sorted(obj_ids[obj_ids != 0])

        #obj_id_locations = {obj_id: anno['objects'][obj_id]['obb']['centroid'] for obj_id in obj_ids}
        #obj_id_dims = {obj_id: anno['objects'][obj_id]['obb']['axesLengths'] for obj_id in obj_ids}

        # get the list of iphone/dslr images and poses
        # NOTE: should be the same as during rasterization
        colmap_camera, image_list, _, distort_params = get_camera_images_poses(scene, cfg.subsample_factor, cfg.image_type)
        # keep first 4 elements
        distort_params = distort_params[:4]

        intrinsic = camera_to_intrinsic(colmap_camera)
        img_height, img_width = colmap_camera.height, colmap_camera.width

        undistort_map1, undistort_map2 = None, None
        if cfg.image_type == 'dslr' and cfg.undistort_dslr:
            undistort_intrinsic = compute_undistort_intrinsic(intrinsic, img_height, img_width, distort_params)
            undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                intrinsic, distort_params, np.eye(3), undistort_intrinsic, (img_width, img_height), cv2.CV_32FC1
            )

        # go through list of images
        for _, image_name in enumerate(tqdm(image_list, desc='image', leave=False)):
            if cfg.image_type == 'iphone':
                image_dir = scene.iphone_rgb_dir
            elif cfg.image_type == 'dslr':
                image_dir = scene.dslr_resized_dir
            # load the image H, W, 3
            img_path = str(image_dir / image_name)
            img = load_image(img_path) 

            rasterout_path = rasterout_dir / scene_id / f'{image_name}.pth'
            raster_out_dict = torch.load(rasterout_path)

            # if dimensions dont match, raster is from downsampled image
            # upsample using nearest neighbor
            pix_to_face = raster_out_dict['pix_to_face'].squeeze().cpu()
            rasterized_dims = list(pix_to_face.shape)

            if rasterized_dims != [img_height, img_width]:
                # upsample pixtoface and zbuf
                pix_to_face = torch.nn.functional.interpolate(pix_to_face.unsqueeze(0).unsqueeze(0).float(),
                                                              size=(img_height, img_width), mode='nearest').squeeze().squeeze().long()
            pix_to_face = pix_to_face.numpy()

            if undistort_map1 is not None and undistort_map2 is not None:
                # apply undistortion to rasterization (nearest neighbor), zbuf (linear) and image (linear)
                pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
                    interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
                )
                # img is np
                img = cv2.remap(img, undistort_map1, undistort_map2,
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            # get object IDs on image
            pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, vtx_obj_ids, mesh)

            if cfg.dbg.viz_obj_ids: # save viz to file
                out_path = viz_obj_ids_dir / f'{image_name}.png'
                viz_ids(img, pix_obj_ids, out_path)

            if cfg.dbg.save_obj_ids: # save viz to file
                out_path = obj_ids_dir / f'{image_name}.png'
                save_ids(img, pix_obj_ids, out_path)



if __name__ == "__main__":
    main()