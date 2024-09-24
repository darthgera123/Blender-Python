import sys
from sys import argv
from argparse import ArgumentParser
from os.path import join
import os 
from shutil import copyfile
import numpy as np

dir = os.path.dirname('/CT/prithvi/work/neural-light-transport/data_2.8/')
if not dir in sys.path:
    sys.path.append(dir)

import bpy

from util import load_json, safe_cast_to_int, normalize_uint, name_from_json_path,\
                img_load, add_b_ch, save_nn, save_uvs
from camera import add_camera, backproject_to_3d
from blender_render import render, render_alpha, easyset
from blender_util import add_light_point
from uv_render import calc_bidir_mapping,calc_light_cosines,calc_view_cosines


def run(args):
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    # Remove existing lights and cameras
    for o in bpy.data.objects:
        if o.type == 'LIGHT' or o.type == 'CAMERA':
            o.select_set(True)
        else:
            o.select_set(False)
    bpy.ops.object.delete()

    camera = args.cam_json
    lights = args.light_json
    
    obj = bpy.data.objects['object']
    cam_name = name_from_json_path(camera)
    light_name = name_from_json_path(lights)
    cam = load_json(camera)
    light = load_json(lights)
    outdir = join(args.output_path,f'{args.type}_{cam_name}_{light_name}')
    os.makedirs(outdir,exist_ok=True)

    # Create scene
    cam_obj = add_camera(
            xyz=cam['position'], rot_vec_rad=cam['rotation'],
            name=cam['name'], f=cam['focal_length'],
            sensor_width=cam['sensor_width'], sensor_height=cam['sensor_height'],
            clip_start=cam['clip_start'], clip_end=cam['clip_end'])
    
    light_obj = add_light_point(xyz=light['position'], name=light['name'], size=light['size'])

    imh = args.imh
    imw = imh / cam['sensor_height'] * cam['sensor_width']
    imw = safe_cast_to_int(imw)
    
    easyset(n_samples=args.spp,color_mode='RGB',h=imh,w=imw)

    # Render rgb image
    rgb_camspc_f = join(outdir, 'rgb_camspc.png')
    render(rgb_camspc_f)
    rgb_camspc = img_load(rgb_camspc_f, as_array=True)[:, :, :3]

    # Render Alpha map
    alpha_f = join(outdir, 'alpha.png')
    render_alpha(alpha_f, samples=args.spp)
    alpha = img_load(alpha_f, as_array=True)
    alpha = normalize_uint(alpha)

    xs, ys = np.meshgrid(range(imw), range(args.imh))
    #    # (0, 0)
    #    # +--------> (w, 0)
    #    # |           x
    #    # |
    #    # v y (0, h)
    xys = np.dstack((xs, ys)).reshape(-1, 2)
    ray_tos, x_locs, x_objnames, x_facei, x_normals = \
        backproject_to_3d(
            xys, cam_obj, obj_names=obj.name, world_coords=True)
    intersect = {
        'ray_tos': ray_tos, 'obj_names': x_objnames, 'face_i': x_facei,
        'locs': x_locs, 'normals': x_normals}
        
    # Compute uv maps in png
    uv2cam, cam2uv = calc_bidir_mapping(
            args.cached_uv_unwrap, obj.name, xys, intersect, args.uvs)
    uv2cam = add_b_ch(uv2cam)
    cam2uv = add_b_ch(cam2uv)
    uv2cam[alpha < 1] = 0

    # Compute light cosines
    lvis_camspc = calc_light_cosines(
            light['position'], xys, intersect, obj)
    
    

    # Compute view cosines
    cvis_camspc = calc_view_cosines(
        cam_obj.location, xys, intersect, obj.name)
    
    # Save different uv maps
    save_uvs(uv2cam,cam2uv,cvis_camspc,lvis_camspc,rgb_camspc,outdir)

    copyfile(camera, join(outdir, 'cam.json'))
    copyfile(lights, join(outdir, 'light.json'))

    # Save neighbours information
    save_nn(args.cam_nn_json,args.light_nn_json,cam_name,light_name,outdir)    

if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    parser = ArgumentParser(description="")
    parser.add_argument(
        '--scene', type=str, required=True, help="path to the .blend scene")
    parser.add_argument(
        '--cached_uv_unwrap', type=str, required=True, help=(
            "path to the cached .pickle of UV unwrapping, which needs doing only "
            "once per scene"))
    parser.add_argument(
        '--cam_json', type=str, required=True, help="path to the camera .json")
    parser.add_argument(
        '--light_json', type=str, required=True, help="path to the light .json")
    parser.add_argument(
        '--cam_nn_json', type=str, required=True,
        help="path to the .json of mapping from camera to its nearest neighbor")
    parser.add_argument(
        '--light_nn_json', type=str, required=True,
        help="path to the .json of mapping from light to its nearest neighbor")
    parser.add_argument(
        '--imh', type=int, default=256,
        help="image height (width derived from camera's sensor height and width)")
    parser.add_argument(
        '--uvs', type=int, default=256, help="size of (square) texture map")
    parser.add_argument(
        '--spp', type=int, default=64, help="samples per pixel for rendering")
    parser.add_argument(
        '--output_path', type=str, required=True, help="output directory")
    parser.add_argument(
        '--type', type=str, required=True, help="trainvali or test")
    parser.add_argument(
        '--debug', type=bool, default=False,
        help="whether to dump additional outputs for debugging")
    run(parser.parse_args(argv))