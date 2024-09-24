import bpy
import os
from os.path import join,dirname
from time import time
from glob import glob
from shutil import move


def easyset(w=None, h=None,
            n_samples=None,
            ao=None,
            color_mode=None,
            file_format=None,
            color_depth=None,
            sampling_method=None,
            n_aa_samples=None):
    """Sets some of the scene attributes more easily.

    Args:
        w (int, optional): Width of render in pixels.
        h (int, optional): Height of render in pixels.
        n_samples (int, optional): Number of samples.
        ao (bool, optional): Ambient occlusion.
        color_mode (str, optional): Color mode of rendering: ``'BW'``,
            ``'RGB'``, or ``'RGBA'``.
        file_format (str, optional): File format of the render: ``'PNG'``,
            ``'OPEN_EXR'``, etc.
        color_depth (str, optional): Color depth of rendering: ``'8'`` or
            ``'16'`` for .png; ``'16'`` or ``'32'`` for .exr.
        sampling_method (str, optional): Method to sample light and
            materials: ``'PATH'`` or ``'BRANCHED_PATH'``.
        n_aa_samples (int, optional): Number of anti-aliasing samples (used
            with ``'BRANCHED_PATH'``).
    """
    scene = bpy.context.scene

    scene.render.resolution_percentage = 100

    if w is not None:
        scene.render.resolution_x = w

    if h is not None:
        scene.render.resolution_y = h

    # Number of samples
    if n_samples is not None and scene.render.engine == 'CYCLES':
        scene.cycles.samples = n_samples

    # Ambient occlusion
    if ao is not None:
        scene.world.light_settings.use_ambient_occlusion = ao

    # Color mode of rendering
    if color_mode is not None:
        scene.render.image_settings.color_mode = color_mode

    # File format of the render
    if file_format is not None:
        scene.render.image_settings.file_format = file_format

    # Color depth of rendering
    if color_depth is not None:
        scene.render.image_settings.color_depth = color_depth

    # Method to sample light and materials
    if sampling_method is not None:
        scene.cycles.progressive = sampling_method

    # Number of anti-aliasing samples
    if n_aa_samples is not None:
        scene.cycles.aa_samples = n_aa_samples

def _render(scene, outnode, result_socket, outpath, exr=True, alpha=True):
    node_tree = scene.node_tree

    # Set output file format
    if exr:
        file_format = 'OPEN_EXR'
        color_depth = '32'
        ext = '.exr'
    else:
        file_format = 'PNG'
        color_depth = '16'
        ext = '.png'
    if alpha:
        color_mode = 'RGBA'
    else:
        color_mode = 'RGB'

    outnode.base_path = '/tmp/%s' % time()

    # Connect result socket(s) to the output node
    if isinstance(result_socket, dict):
        assert exr, ".exr must be used for multi-layer results"
        file_format += '_MULTILAYER'

        assert 'composite' in result_socket.keys(), \
            ("Composite pass is always rendered anyways. Plus, we need this "
             "dummy connection for the multi-layer OpenEXR file to be saved "
             "to disk (strangely)")
        node_tree.links.new(result_socket['composite'],
                            outnode.inputs['Image'])

        # Add input slots and connect
        for k, v in result_socket.items():
            outnode.layer_slots.new(k)
            node_tree.links.new(v, outnode.inputs[k])

        render_f = join(outnode.base_path, '????.exr')
    else:
        node_tree.links.new(result_socket, outnode.inputs['Image'])

        render_f = join(outnode.base_path, 'Image????' + ext)

    outnode.format.file_format = file_format
    outnode.format.color_depth = color_depth
    outnode.format.color_mode = color_mode

    scene.render.filepath = '/tmp/%s' % time() # composite (to discard)

    # Render
    bpy.ops.render.render(write_still=True)

    # Depending on the scene state, the render filename may be anything
    # matching the pattern
    fs = glob(render_f)
    assert len(fs) == 1, \
        ("There should be only one file matching:\n\t{p}\n"
         "but found {n}").format(p=render_f, n=len(fs))
    render_f = fs[0]

    # Move from temporary directory to the desired location
    if not outpath.endswith(ext):
        outpath += ext
    move(render_f, outpath)
    return outpath

def _render_prepare(cam, obj_names):
    if cam is None:
        cams = [o for o in bpy.data.objects if o.type == 'CAMERA']
        assert (len(cams) == 1), \
            "With `cam` not provided, there must be exactly one camera"
        cam = cams[0]

    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in bpy.data.objects if o.type == 'MESH']
    # Should be a list of strings by now

    for x in obj_names:
        assert isinstance(x, str), \
            ("Objects should be specified by their names (strings), not "
             "objects themselves")

    scene = bpy.context.scene

    # Set active camera
    scene.camera = cam

    # Hide objects to ignore
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = obj.name not in obj_names

    scene.use_nodes = True

    # Clear the current scene node tree to avoid unexpected renderings
    nodes = scene.node_tree.nodes
    for n in nodes:
        if n.name != "Render Layers":
            nodes.remove(n)

    outnode = nodes.new('CompositorNodeOutputFile')

    return cam.name, obj_names, scene, outnode


def render(outpath, cam=None, obj_names=None, alpha=True, text=None):
    """Renders current scene with cameras in scene.

    Args:
        outpath (str): Path to save the render to. Should end with either
            .exr or .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, use the only camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. If ``None``, all objects are of interest and will
            appear in the render.
        alpha (bool, optional): Whether to render the alpha channel.
        text (dict, optional): What text to be overlaid on image and how,
            following the format::

                {
                    'contents': "Hello World!",
                    'bottom_left_corner': (50, 50),
                    'font_scale': 1,
                    'bgr': (255, 0, 0),
                    'thickness': 2,
                }

    Writes
        - A 32-bit .exr or 16-bit .png image.
    """
    outdir = dirname(outpath)
    os.makedirs(outdir,exist_ok=True)
    
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    scene.render.film_transparent = False
    result_socket = scene.node_tree.nodes['Render Layers'].outputs['Image']

    # Render
    exr = outpath.endswith('.exr')
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=exr, alpha=alpha)


    print(f"{obj_names} rendered through {cam_name}")


def render_envmap(outpath, cam=None, obj_names=None, alpha=True, text=None,envmap=None):
    """Renders current scene with cameras in scene.

    Args:
        outpath (str): Path to save the render to. Should end with either
            .exr or .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, use the only camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. If ``None``, all objects are of interest and will
            appear in the render.
        alpha (bool, optional): Whether to render the alpha channel.
        text (dict, optional): What text to be overlaid on image and how,
            following the format::

                {
                    'contents': "Hello World!",
                    'bottom_left_corner': (50, 50),
                    'font_scale': 1,
                    'bgr': (255, 0, 0),
                    'thickness': 2,
                }

    Writes
        - A 32-bit .exr or 16-bit .png image.
    """
    outdir = dirname(outpath)
    os.makedirs(outdir,exist_ok=True)
    

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    scene.render.engine = 'CYCLES'
    scene.render.film_transparent = False
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    
    
    # wd = scene.world
    # nt = bpy.data.worlds[wd.name].node_tree
    # backNode = node_tree.nodes['Environment Texture']
    # backNode.image = bpy.data.images.load(envmap)
    
    # world = scene.world
    # Link the Environment Texture node to the World's background shader
    
    
    # Load your HDRI image
    hdri_image_path = envmap
    # world.light_settings.environment_map = bpy.data.images.load(hdri_image_path)

    # Create a ShaderNodeBackground and link it to the Environment Texture
    env_texture_node = scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env_texture_node.image = bpy.data.images.load(hdri_image_path)

    background_node = scene.world.node_tree.nodes.new("ShaderNodeBackground")
    scene.world.node_tree.links.new(
        background_node.inputs["Color"], 
        env_texture_node.outputs["Color"]
    )
    # scene.world.node_tree.links.new(
    #     outnode, 
    #     env_texture_node.outputs["Color"]
    # )
    
    # Render
    exr = outpath.endswith('.exr')
    result_socket = scene.node_tree.nodes['Render Layers'].outputs['Image']
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=exr, alpha=alpha)


    print(f"{obj_names} rendered through {cam_name}")



def render_alpha(outpath, cam=None, obj_names=None, samples=1000):
    r"""Renders binary or soft mask of objects from the specified camera.

    Args:
        outpath (str): Path to save the render to. Should end with .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be just one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        samples (int, optional): Samples per pixel. :math:`1` gives a hard
            mask, and :math:`\gt 1` gives a soft (anti-aliased) mask.

    Writes
        - A 16-bit three-channel .png mask, where bright indicates
          foreground.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    scene.render.engine = 'CYCLES'
    film_transparent_old = scene.render.film_transparent
    scene.render.film_transparent = True
    # Anti-aliased edges are built up by averaging multiple samples
    samples_old = scene.cycles.samples
    scene.cycles.samples = samples

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    result_socket = nodes['Render Layers'].outputs['Alpha']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=False, alpha=False)

    # Restore
    scene.cycles.samples = samples_old
    scene.cycles.film_transparent = film_transparent_old

    print(
        f"Foreground alpha of {obj_names} rendered through {cam_name}")

def render_visibility(outpath, cam=None, obj_names=None, samples=1000,exr=False):
    r"""Renders binary or soft mask of objects from the specified camera.

    Args:
        outpath (str): Path to save the render to. Should end with .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be just one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        samples (int, optional): Samples per pixel. :math:`1` gives a hard
            mask, and :math:`\gt 1` gives a soft (anti-aliased) mask.

    Writes
        - A 16-bit three-channel .png mask, where bright indicates
          foreground.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    scene.view_layers["View Layer"].use_pass_shadow = True
    scene.render.engine = 'CYCLES'
    
    # Anti-aliased edges are built up by averaging multiple samples
    samples_old = scene.cycles.samples
    scene.cycles.samples = samples

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    # shadow_socket = nodes['Render Layers'].outputs['Shadow']
    
    greater_than_node = nodes.new(type='CompositorNodeMath')
    greater_than_node.operation = 'GREATER_THAN'

    greater_than_node.inputs[1].default_value = 0.0

    # normalize_node = nodes.new(type='CompositorNodeNormalize')

    # color_ramp_node = nodes.new(type='CompositorNodeValToRGB')

    # color_ramp_node.color_ramp.elements[0].position = 0.0
    # color_ramp_node.color_ramp.elements[1].position = 1.0

    # # color_ramp_node = color_ramp_node.outputs['Image']

    # node_tree.links.new(shadow_socket, greater_than_node.inputs[0])

    # # Connect the Greater Than node output to the Normalize node input
    # node_tree.links.new(greater_than_node.outputs[0], normalize_node.inputs[0])

    # # Connect the Normalize node output to the Color Ramp node input
    # node_tree.links.new(normalize_node.outputs[0], color_ramp_node.inputs[0])

    # result_socket = color_ramp_node
    result_socket = nodes['Render Layers'].outputs['Shadow']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=exr, alpha=False)
    
    

    print(
        f"Foreground alpha of {obj_names} rendered through {cam_name}")

def render_normal(outpath, cam=None, obj_names=None, samples=1000,exr=False):
    r"""Renders normal of objects from the specified camera.

    Args:
        outpath (str): Path to save the render to. Should end with .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be just one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        samples (int, optional): Samples per pixel. :math:`1` gives a hard
            mask, and :math:`\gt 1` gives a soft (anti-aliased) mask.

    Writes
        - A 16-bit three-channel .png mask, where bright indicates
          foreground.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.render.engine = 'CYCLES'
    
    # Anti-aliased edges are built up by averaging multiple samples
    samples_old = scene.cycles.samples
    scene.cycles.samples = samples

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    # shadow_socket = nodes['Render Layers'].outputs['Shadow']
    
    
    # normalize_node = nodes.new(type='CompositorNodeNormalize')

    # color_ramp_node = nodes.new(type='CompositorNodeValToRGB')

    # color_ramp_node.color_ramp.elements[0].position = 0.0
    # color_ramp_node.color_ramp.elements[1].position = 1.0

    # # color_ramp_node = color_ramp_node.outputs['Image']

    # node_tree.links.new(shadow_socket, greater_than_node.inputs[0])

    # # Connect the Greater Than node output to the Normalize node input
    # node_tree.links.new(greater_than_node.outputs[0], normalize_node.inputs[0])

    # # Connect the Normalize node output to the Color Ramp node input
    # node_tree.links.new(normalize_node.outputs[0], color_ramp_node.inputs[0])

    # result_socket = color_ramp_node
    result_socket = nodes['Render Layers'].outputs['Normal']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=exr, alpha=False)
    
    

    print(
        f"Foreground alpha of {obj_names} rendered through {cam_name}")
    
def render_diffuse(outpath, cam=None, obj_names=None, samples=1000,exr=False):
    r"""Renders normal of objects from the specified camera.

    Args:
        outpath (str): Path to save the render to. Should end with .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be just one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        samples (int, optional): Samples per pixel. :math:`1` gives a hard
            mask, and :math:`\gt 1` gives a soft (anti-aliased) mask.

    Writes
        - A 16-bit three-channel .png mask, where bright indicates
          foreground.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.render.engine = 'CYCLES'
    
    # Anti-aliased edges are built up by averaging multiple samples
    samples_old = scene.cycles.samples
    scene.cycles.samples = samples

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    # shadow_socket = nodes['Render Layers'].outputs['Shadow']
    
    
    # normalize_node = nodes.new(type='CompositorNodeNormalize')

    # color_ramp_node = nodes.new(type='CompositorNodeValToRGB')

    # color_ramp_node.color_ramp.elements[0].position = 0.0
    # color_ramp_node.color_ramp.elements[1].position = 1.0

    # # color_ramp_node = color_ramp_node.outputs['Image']

    # node_tree.links.new(shadow_socket, greater_than_node.inputs[0])

    # # Connect the Greater Than node output to the Normalize node input
    # node_tree.links.new(greater_than_node.outputs[0], normalize_node.inputs[0])

    # # Connect the Normalize node output to the Color Ramp node input
    # node_tree.links.new(normalize_node.outputs[0], color_ramp_node.inputs[0])

    # result_socket = color_ramp_node
    result_socket = nodes['Render Layers'].outputs['DiffCol']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=exr, alpha=False)
    
    

    print(
        f"Foreground alpha of {obj_names} rendered through {cam_name}")

def render_position(outpath, cam=None, obj_names=None, samples=1000,exr=False):
    r"""Renders normal of objects from the specified camera.

    Args:
        outpath (str): Path to save the render to. Should end with .png.
        cam (bpy_types.Object, optional): Camera through which scene is
            rendered. If ``None``, there must be just one camera in scene.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means all objects.
        samples (int, optional): Samples per pixel. :math:`1` gives a hard
            mask, and :math:`\gt 1` gives a soft (anti-aliased) mask.

    Writes
        - A 16-bit three-channel .png mask, where bright indicates
          foreground.
    """
    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)
    scene.view_layers["View Layer"].use_pass_position = True
    scene.render.engine = 'CYCLES'
    
    # Anti-aliased edges are built up by averaging multiple samples
    samples_old = scene.cycles.samples
    scene.cycles.samples = samples

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    # shadow_socket = nodes['Render Layers'].outputs['Shadow']
    
    
    # normalize_node = nodes.new(type='CompositorNodeNormalize')

    # color_ramp_node = nodes.new(type='CompositorNodeValToRGB')

    # color_ramp_node.color_ramp.elements[0].position = 0.0
    # color_ramp_node.color_ramp.elements[1].position = 1.0

    # # color_ramp_node = color_ramp_node.outputs['Image']

    # node_tree.links.new(shadow_socket, greater_than_node.inputs[0])

    # # Connect the Greater Than node output to the Normalize node input
    # node_tree.links.new(greater_than_node.outputs[0], normalize_node.inputs[0])

    # # Connect the Normalize node output to the Color Ramp node input
    # node_tree.links.new(normalize_node.outputs[0], color_ramp_node.inputs[0])

    # result_socket = color_ramp_node
    result_socket = nodes['Render Layers'].outputs['Position']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath,
                      exr=exr, alpha=False)
    
    

    print(
        f"Foreground alpha of {obj_names} rendered through {cam_name}")