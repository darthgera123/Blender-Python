# Blender Python Scripts for Rendering and Visualization

This repository contains several Python scripts tailored for use with Blender 2.83. These scripts provide functionalities for camera and light setup, rendering scenes, batch processing, and visualization. Below is a brief description of each script.

## Files

### 1. `blender_render.py`
This script handles the rendering operations in Blender, allowing you to:

- **Configure Scene Settings**: Set render resolution, sample count, color mode, and file format using the `easyset` function.
- **Render with Camera**: Render scenes using camera views and save outputs as PNG or EXR files using `render`, `render_envmap`, `render_alpha`, and other functions.
- **Advanced Passes**: Render visibility, normal, diffuse, and position passes for selected objects.

### 2. `blender_vis.py`
Provides utility functions to manage camera and lighting in Blender:

- **Camera Setup**: Add and configure cameras in the scene using `add_camera` and `easyset`.
- **Light Setup**: Add and configure point lights using `add_light_point`.
- **Camera and Light Management**: Load camera and light settings from JSON files and apply them to the scene.

### 3. `claudia_olat_psnerf.py`
Designed for batch processing of images with a focus on OLAT (One Light At a Time) renderings:

- **Batch Image Processing**: Processes images from multiple camera and light configurations, converts them to PNG format, and saves them.
- **I/O Operations**: Load and save image files in EXR and PNG formats, with additional JSON data handling.

### 4. `avg_img.py`
Averages multiple frames of video or images from multiple lighting conditions and cameras:

- **Extract Frames**: Extracts frames from videos or images, processes them, and averages the lighting over multiple images to generate mean images.
- **Save Images**: Saves the processed images in PNG or JPG format.

### 5. `background.py`
Handles scene creation and rendering with a background and specific camera settings:

- **Background Setup**: Loads cameras from JSON files and adds background lighting using an environment map.
- **Scene Rendering**: Uses the `render` function to save the final output image for each camera.

### 6. `dill_light.py`
Manages lighting for different views and renders with specific camera settings:

- **Light Setup**: Adds lights to the scene and controls their parameters.
- **Camera Rendering**: Renders visibility, normal, diffuse, and other passes for the scene.

### 7. `normal_psnerf.py`
Processes images from different cameras and lights, normalizing and saving them:

- **Normalize Images**: Loads EXR images, normalizes their pixel values, and saves them as PNG images.
- **Batch Processing**: Processes multiple images from different camera views and lighting conditions.

### 8. `psnerf_data.py`
Extracts frames from video data and processes them for further use:

- **Frame Extraction**: Extracts and processes frames from video files.
- **Save Processed Images**: Saves the processed frames as PNG images, organizing them by view and lighting condition.

### 9. `read_cam.py`
Manages camera loading and manipulation in Blender:

- **Camera Loading**: Loads camera configurations from JSON files.
- **Render Settings**: Uses the `easyset` function to adjust render parameters like resolution and sample count.

### 10. `rotate_background.py`
Similar to `background.py`, this script rotates the background environment map and renders the scene from different camera angles.

### 11. `vis_psnerf.py`
Handles visualization of images captured under different lighting conditions:

- **Image Processing**: Processes images from multiple lighting conditions, visualizing them as a combined result.
- **Batch Processing**: Loads and saves processed images for each camera view.

### 12. `animate_people.py`
Handles animation of people models using predefined motion paths:

- **Bone Animation**: Rotates and translates bones based on input motion data.
- **Keyframe Insertion**: Adds keyframes for bone rotations and translations.

### 13. `camera.py`
Provides utilities for adding and configuring cameras in Blender:

- **Add Camera**: Adds a camera to the scene and configures its settings.
- **Camera Calibration**: Computes the intrinsic and extrinsic matrices for camera calibration.

### 14. `check_empty.py`
Checks for empty directories and deletes them:

- **Directory Deletion**: Deletes empty directories in a given path.

### 15. `claudia_nerf_data.py`
Loads NeRF data and processes it for rendering with Blender:

- **Light and Camera Setup**: Loads light and camera configurations from JSON files.
- **Render NeRF Data**: Renders NeRF data using the configured lights and cameras.

### 16. `img2gif.py`
Converts a sequence of images into a GIF:

- **GIF Creation**: Loads images from a directory and saves them as a GIF.

### 17. `uv_render.py`
Handles UV rendering and mapping in Blender:

- **UV Mapping**: Processes UV coordinates and interpolates values onto a UV canvas.

### 18. `uv_unwrap.py`
Handles UV unwrapping for 3D objects:

- **UV Unwrapping**: Unwraps UV maps for a given 3D object.

## Installation

To run this code, you should use the Python bundled inside Blender 2.83, rather than your system or environment Python. Here's how to set it up:

1. Clone this repository:

    ```bash
    cd "$ROOT"
    git clone https://github.com/google/neural-light-transport.git
    ```

2. Install Blender-Python (the binaries are pre-built, so just download and unzip):

    ```bash
    cd "$WHERE_YOU_WANT_BLENDER_INSTALLED"

    # Download Blender 2.83
    wget https://download.blender.org/release/Blender2.83/blender-2.83-linux-x64.tar.bz2

    # Unzip the pre-built binaries
    tar -xvjf blender-2.83-linux-x64.tar.bz2
    ```

3. Install the dependencies for Blender's bundled Python:

    ```bash
    cd blender-2.83-linux-x64/2.83/python/bin

    # Install pip for THIS Blender-bundled Python
    curl https://bootstrap.pypa.io/get-pip.py | ./python3.7m

    # Use THIS pip to install other dependencies
    ./pip install Pillow
    ./pip install tqdm
    ./pip install ipython
    ./pip install numpy
    ./pip install opencv-python
    ```

4. Make sure this Python can locate `xiuminglib`:

    ```bash
    export PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH
    ```

## Run Commands

You can run the scripts using Blender with the following commands:

1. **Rotate Background Rendering**:
    ```bash
    $BLENDER --background --python rotate_background.py -- \
    --scene=<scene.blend> \
    --output_path=<output_path> \
    --env=<env_folder> \
    --envmap=<envmap_name> \
    --cam_json=<cam_json>
    ```

2. **Background Rendering**:
    ```bash
    $BLENDER --background --python background.py -- \
    --scene=<scene.blend> \
    --output_path=<output_path> \
    --env=<env_folder> \
    --envmap=<envmap_name> \
    --cam_json=<cam_json>
    ```

3. **Dill Light Rendering with Multiple Outputs**:
    ```bash
    $BLENDER --background --python dill_light.py -- \
    --scene=<scene.blend> \
    --rgb_video --mask --rgb_exr \
    --output_path=<output_path>
    ```

Now you are ready to run the scripts using Blender 2.83 and its bundled Python.
