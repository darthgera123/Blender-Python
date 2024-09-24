from sys import argv
from argparse import ArgumentParser
import pickle as pkl
import bpy
import numpy as np
import os 

parser = ArgumentParser()
parser.add_argument('--scene', type=str, default='')
parser.add_argument('--object_name', type=str, default='')
parser.add_argument(
'--angle_limit', type=float, default=89., help=(
    "angle limit; lower for more projection groups, and higher for less "
    "distortion"))
parser.add_argument(
'--area_weight', type=float, default=1., help=(
    "area weight used to weight projection vectors; higher for fewer "
    "islands"))
parser.add_argument('--outpath', type=str, required=True, help="output .pickle")

def has_uv_map(obj):
    return len(obj.data.uv_layers) > 0

def uv_unwrap(obj,name,angle_limit,area_weight):
    assert obj.type == 'MESH'
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if not has_uv_map(obj):
        print("UV map not present")
        bpy.ops.uv.smart_project(
            angle_limit=angle_limit, user_area_weight=area_weight)
    else:
        print("UV map there")
    bpy.ops.object.mode_set(mode='OBJECT')
    fi_li_vi_u_v = {}
    for f in obj.data.polygons:
        li_vi_u_v = []
        for vi, li in zip(f.vertices, f.loop_indices):
            uv = obj.data.uv_layers.active.data[li].uv
            li_vi_u_v.append([li, vi, uv.x, uv.y])
        fi_li_vi_u_v[f.index] = np.array(li_vi_u_v)

    # bpy.ops.uv.image_editor_enter()
    # bpy.ops.uv.export_layout(filepath=name,
    #                         mode='PNG',
    #                         check_existing=True,                            )

    return fi_li_vi_u_v


def run(args):
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    obj = bpy.data.objects[args.object_name]
    name=args.outpath.split('.')[0]+'.png'

    fi_li_vi_u_v = uv_unwrap(obj,name,angle_limit=args.angle_limit,\
                    area_weight=args.area_weight)
    
    
    with open(args.outpath,'wb') as f:
        pkl.dump(fi_li_vi_u_v,f)


if __name__ == '__main__':
    # os.makedirs(args.output_dir, exist_ok=True)
    # open a blend file
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    run(parser.parse_args(argv))
