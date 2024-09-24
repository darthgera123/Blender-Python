import bpy
import bmesh
from os.path import basename

def add_light_point(
        xyz=(0, 0, 0), name=None, size=0, color=(1, 1, 1), energy=100):
    
    bpy.ops.object.light_add(type='POINT', location=xyz)
    point = bpy.context.active_object
    
    if name is not None:
        point.name = name

    if len(color) == 3:
        color += (1.,)

    point.data.shadow_soft_size = size

    # Strength
    engine = bpy.context.scene.render.engine
    
    if engine == 'CYCLES':
        point.data.use_nodes = True
        point.data.node_tree.nodes['Emission'].inputs[
            'Strength'].default_value = energy
        point.data.node_tree.nodes['Emission'].inputs[
            'Color'].default_value = color
        
    else:
        raise NotImplementedError(engine)

    print(f"Light {point.name} added")

    return point

def add_light_env(env=(1, 1, 1, 1), strength=1, rot_vec_rad=(0, 0, 0),
                  scale=(1, 1, 1)):
    r"""Adds environment lighting.

    Args:
        env (tuple(float) or str, optional): Environment map. If tuple,
            it's RGB or RGBA, each element of which :math:`\in [0,1]`.
            Otherwise, it's the path to an image.
        strength (float, optional): Light intensity.
        rot_vec_rad (tuple(float), optional): Rotations in radians around x,
            y and z.
        scale (tuple(float), optional): If all changed simultaneously,
            then no effects.
    """
    engine = bpy.context.scene.render.engine
    assert engine == 'CYCLES', "Rendering engine is not Cycles"

    if isinstance(env, str):
        bpy.data.images.load(env, check_existing=True)
        env = bpy.data.images[basename(env)]
    else:
        if len(env) == 3:
            env += (1,)
        assert len(env) == 4, "If tuple, env must be of length 3 or 4"

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new('ShaderNodeBackground')
    links.new(bg_node.outputs['Background'],
              nodes['World Output'].inputs['Surface'])

    if isinstance(env, tuple):
        # Color
        bg_node.inputs['Color'].default_value = env
        print(("Environment is pure color, "
                        "so rotation and scale have no effect"))
    else:
        # Environment map
        texcoord_node = nodes.new('ShaderNodeTexCoord')
        env_node = nodes.new('ShaderNodeTexEnvironment')
        env_node.image = env
        mapping_node = nodes.new('ShaderNodeMapping')
        # mapping_node.rotation = rot_vec_rad
        # mapping_node.scale = scale
        mapping_node.inputs[2].default_value = rot_vec_rad  # Rotation (Euler) as a 3D vector
        mapping_node.inputs[3].default_value = scale     
        links.new(texcoord_node.outputs['Generated'],
                  mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], env_node.inputs['Vector'])
        links.new(env_node.outputs['Color'], bg_node.inputs['Color'])

    bg_node.inputs['Strength'].default_value = strength
    print("Environment light added")




def get_bmesh(obj):
    """Gets Blender mesh data from object.

    Args:
        obj (bpy_types.Object): Object.

    Returns:
        BMesh: Blender mesh data.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.view_layer.update()

    return bm

def raycast(obj_bvhtree, ray_from_objspc, ray_to_objspc):
    """Casts a ray to an object.

    Args:
        obj_bvhtree (mathutils.bvhtree.BVHTree): Constructed BVH tree of the
            object.
        ray_from_objspc (mathutils.Vector): Ray origin, in object's local
            coordinates.
        ray_to_objspc (mathutils.Vector): Ray goes through this point, also
            specified in the object's local coordinates. Note that the ray
            doesn't stop at this point, and this is just for computing the
            ray direction.

    Returns:
        tuple:
            - **hit_loc** (*mathutils.Vector*) -- Hit location on the object,
              in the object's local coordinates. ``None`` means no
              intersection.
            - **hit_normal** (*mathutils.Vector*) -- Normal of the hit
              location, also in the object's local coordinates.
            - **hit_fi** (*int*) -- Index of the face where the hit happens.
            - **ray_dist** (*float*) -- Distance that the ray has traveled
              before hitting the object. If ``ray_to_objspc`` is a point on
              the object surface, then this return value is useful for
              checking for self occlusion.
    """
    ray_dir = (ray_to_objspc - ray_from_objspc).normalized()
    hit_loc, hit_normal, hit_fi, ray_dist = \
        obj_bvhtree.ray_cast(ray_from_objspc, ray_dir)
    if hit_loc is None:
        assert hit_normal is None and hit_fi is None and ray_dist is None
    return hit_loc, hit_normal, hit_fi, ray_dist