# -*- coding: utf-8 -*-
import pybullet as p
import numpy as np
import cv2

# Constants from geom.py, to be centralized later if needed
TABLE_TOP_Z = 0.65
CAMERA_TARGET = [0.5, 0, TABLE_TOP_Z]
CAMERA_DISTANCE = 1.2
CAMERA_PARAMS = {
    'width': 224,
    'height': 224, 
    'fov': 60.0,
    'near': 0.1,
    'far': 2.0
}

def set_topdown_camera(target=CAMERA_TARGET, distance=CAMERA_DISTANCE, 
                       yaw=90.0, pitch=-89.0, **camera_params):
    """Sets up a top-down camera and returns its parameters."""
    params = {**CAMERA_PARAMS, **camera_params}
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        params['fov'], 
        params['width']/float(params['height']), 
        params['near'], 
        params['far']
    )
    return params['width'], params['height'], view_matrix, proj_matrix

def get_rgb_depth(width, height, view, proj):
    """Captures RGB and depth images from the simulation."""
    img_arr = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(img_arr[2], dtype=np.uint8).reshape(height, width, 4)[..., :3]
    depth = np.asarray(img_arr[3], dtype=np.float32).reshape(height, width)
    return rgb, depth

def find_best_grasp_pixel(rgb, depth):
    """Finds the best pixel to grasp based on depth and color information."""
    height, width = rgb.shape[:2]
    
    # Use depth to find foreground objects
    foreground_mask = (depth > 0) & (depth < depth.max() * 0.95)
    
    if np.sum(foreground_mask) < 100: # If not enough points, use color
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        foreground_mask = gray < 200

    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return width // 2, height // 2 # Default to center

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return width // 2, height // 2

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return cx, cy

def pixel_to_world(u, v, depth_value, view_matrix, proj_matrix):
    """Converts a pixel coordinate to a world coordinate."""
    width, height = CAMERA_PARAMS['width'], CAMERA_PARAMS['height']
    x_ndc = (2.0 * u / width) - 1.0
    y_ndc = 1.0 - (2.0 * v / height)
    
    # The depth value from the depth buffer is in range [0, 1]
    # We need to convert it to a Z coordinate in clip space
    z_clip = depth_value * 2.0 - 1.0

    # From clip space to view space
    proj_matrix_np = np.array(proj_matrix).reshape(4, 4)
    inv_proj = np.linalg.inv(proj_matrix_np)
    pos_view = inv_proj @ np.array([x_ndc, y_ndc, z_clip, 1.0])
    pos_view /= pos_view[3]

    # From view space to world space
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    inv_view = np.linalg.inv(view_matrix_np)
    pos_world = inv_view @ pos_view
    
    return pos_world[:3]
