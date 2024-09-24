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
from blender_render import render, render_alpha, easyset, render_visibility, render_normal, render_diffuse, render_position
from blender_util import add_light_point


def create_mat(path):
    mat = bpy.data.materials.new(name="Normal")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    # create normal
    normal = mat.node_tree.nodes.new('ShaderNodeNormalMap')
    mat.node_tree.links.new(bsdf.inputs['Normal'], normal.outputs['Normal'])
    return mat 

def read_numbers_from_file(filename):                          
     numbers_list = []                                          
     with open(filename, 'r') as file:                          
             for line in file:                          
                     numbers = line.strip().split()     
                     numbers = [float(n) for n in numbers]        
                     numbers_list.append(numbers)          
     return numbers_list

def swap_first_two_elements(lst):
    if len(lst) >= 2:
        lst[1], lst[2] = -lst[2], lst[1]
    return lst

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
    # lights = args.light_json
    
    # image_tex = args.image_tex
    # normal_tex = args.normal_tex
    # #  for appearance, choose mesh
    # # for pose choose armature
    
    # # obj = bpy.data.objects['object']
    # obj_mesh = bpy.data.objects['object']
    # obj_arm = bpy.data.objects['object.001'] # obj.type==armature
    
    cam_name = name_from_json_path(camera)
    # light_name = name_from_json_path(lights)
    cam = load_json(camera)
    light = read_numbers_from_file('/CT/prithvi2/static00/DiLiGenT-MV/mvpmsData/bearPNG/view_01/light_directions.txt')
    # light = load_json(lights)
    # all_lights=['L030.json','L069.json','L058.json','L063.json']
    # lights = '/CT/prithvi/static00/synth_light_stage/lights/'
    # for l in all_lights:
    #     light = load_json(os.path.join(lights,l))
    #     light_obj = add_light_point(xyz=light['position'], name=light['name'], size=light['size'],energy=10)

    # light = load_json(lights)
    # outdir = join(args.output_path,f'{cam_name}_{light_name}')
    # outdir = join(args.output_path,f'{cam_name}')
    

    # Create scene
    cam_obj = add_camera(
            xyz=cam['position'], rot_vec_rad=cam['rotation'],
            name=cam['name'], f=cam['focal_length'],
            sensor_width=cam['sensor_width'], sensor_height=cam['sensor_height'],
            clip_start=cam['clip_start'], clip_end=cam['clip_end'])
    
    # light_obj = add_light_point(xyz=light['position'], name=light['name'], size=light['size'])

    
    imh = 512
    imw = imh / cam['sensor_height'] * cam['sensor_width']
    imw = safe_cast_to_int(imw)
    
    easyset(n_samples=256,color_mode='RGB',h=imh,w=imw)


    # initialize
    bpy.context.scene.frame_end = 100

    

# # # # Render the animation
    bpy.context.scene.camera=cam_obj
    start_frame = 1
    end_frame = 100
    for num,l in enumerate(light):
        k = swap_first_two_elements(l)
        print(k)
        light_obj = add_light_point(xyz=k, size=0.1,energy=2.5)
        light_name = f'L{str(num).zfill(3)}'
        if args.rgb_video:
            
            outdir = join(args.output_path,f'{cam_name}_{light_name}')
            os.makedirs(outdir,exist_ok=True)
            # bpy.context.scene.render.filepath = join(outdir, 'rgb_camspc_rot.mp4')
            # bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
            # bpy.ops.render.render(animation=True)
            for i in range(start_frame, end_frame + 1):
            # Set the current frame
                bpy.context.scene.frame_set(i)
            # Set the filepath to save the current frame to
            # bpy.data.scenes['Scene'].render.filepath = outdir + 'frame_' + str(i).zfill(4)
            # Render and save the current frame
                rgb_camspc_f = join(outdir, f'{str(i-1).zfill(6)}.png')
                render(rgb_camspc_f)
        if args.visibility:
            visible_dir = join(args.output_path,'visibility',f'{cam_name}_{light_name}')
            os.makedirs(visible_dir,exist_ok=True)
            for i in range(start_frame, end_frame + 1):
                bpy.context.scene.frame_set(i)
                visible = join(visible_dir, f'{str(i-1).zfill(6)}.exr')
                render_visibility(visible, samples=256,exr=True)
        

        
        
        
        
        for o in bpy.data.objects:
            if o.type == 'LIGHT':
                o.select_set(True)
            else:
                o.select_set(False)
        bpy.ops.object.delete()
    
    if args.normal:
        normal_dir = join(args.output_path,'normal',f'{cam_name}')
        os.makedirs(normal_dir,exist_ok=True)
        for i in range(start_frame, end_frame + 1):
            bpy.context.scene.frame_set(i)
            normal = join(normal_dir, f'{str(i-1).zfill(6)}')
            render_normal(normal, samples=256,exr=True)
    if args.diffuse:
        diffuse_dir = join(args.output_path,'diffuse',f'{cam_name}')
        os.makedirs(diffuse_dir,exist_ok=True)
        for i in range(start_frame, end_frame + 1):
            bpy.context.scene.frame_set(i)
            diffuse = join(diffuse_dir, f'{str(i-1).zfill(6)}')
            render_diffuse(diffuse, samples=256,exr=True)
    if args.position:
        position_dir = join(args.output_path,'position',f'{cam_name}')
        os.makedirs(position_dir,exist_ok=True)
        for i in range(start_frame, end_frame + 1):
            bpy.context.scene.frame_set(i)
            position = join(position_dir, f'{str(i-1).zfill(6)}')
            render_position(position, samples=256,exr=True)
    
    # if args.mask:
        
    #     # alpha_dir = join(args.output_path,'mask_cihp',f'{cam_name}_{light_name}')
    #     alpha_dir = join(args.output_path,'mask_cihp',f'{cam_name}')
    #     # if os.path.isdir(alpha_dir):
    #     #     os.makedirs(alpha_dir)
    #     os.makedirs(alpha_dir,exist_ok=True)
    #     for i in range(start_frame, end_frame + 1):
    #         bpy.context.scene.frame_set(i)
    #         alpha_f = join(alpha_dir, f'{str(i-1).zfill(6)}.png')
    #         render_alpha(alpha_f, samples=256)
    # if args.visibility:
    #     visible_dir = join(args.output_path,'visibility',f'{cam_name}_{light_name}')
    #     os.makedirs(visible_dir,exist_ok=True)
    #     for i in range(start_frame, end_frame + 1):
    #         bpy.context.scene.frame_set(i)
    #         visible = join(visible_dir, f'{str(i-1).zfill(6)}.exr')
    #         render_visibility(visible, samples=256,exr=True)
    # rgb_camspc_f = join(outdir, 'rgb_camspc_rot1.png')
    # render(rgb_camspc_f)

    # for i in range(start_frame, end_frame + 1):
        # Set the current frame
        
        # Set the filepath to save the current frame to
        # bpy.data.scenes['Scene'].render.filepath = outdir + 'frame_' + str(i).zfill(4)
    #     # Render and save the current frame
    #     # rgb_camspc_f = join(outdir, f'{str(i).zfill(6)}.png')
    #     # render(rgb_camspc_f)
    
    # # # Render Alpha map
    
    
    

    # Render rgb image
    # obj.data.materials[0] = create_mat(image_tex)
    # rgb_camspc_f = join(outdir, 'rgb_camspc_rot1.png')
    # render(rgb_camspc_f)
    
    # # Render Alpha map
    # alpha_f = join(outdir, 'alpha.png')
    # render_alpha(alpha_f, samples=args.spp)
    
    # obj.data.materials[0] = create_mat(normal_tex)
    # rgb_camspc_f = join(outdir, 'normal_camspc.png')
    # render(rgb_camspc_f)
    

if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    parser = ArgumentParser(description="")
    parser.add_argument(
        '--scene', type=str, required=True, help="path to the .blend scene")
    parser.add_argument(
        '--cam_json', type=str, required=True, help="path to the camera .json")
    parser.add_argument(
        '--light_json', type=str, help="path to the light .json")
    parser.add_argument(
        '--output_path', type=str, required=True, help="output directory")
    parser.add_argument(
        '--rgb_video', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--mask', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--visibility', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--normal', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--diffuse', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--position', action='store_true',  help="save_rgb_video")
    run(parser.parse_args(argv))