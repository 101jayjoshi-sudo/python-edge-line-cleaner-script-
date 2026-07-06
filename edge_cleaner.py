#!/usr/bin/env python3
"""
Python Edge-Line Cleaner Script
===============================

This script scans PNG image files, detects black/dark pixels connected to the 
outer frame of the image, and erases those pixels (the "leads" or connection lines)
while leaving all interior marks (symbols, text, boxes, circles, fine lines) fully intact.

It leverages a hybrid skeletonization and distance-transform-guided erasure algorithm, 
followed by OpenCV's inpainting to cleanly restore symbol borders where lines intersected.

Dependencies:
-------------
- numpy
- opencv-python (cv2)
- scipy
- scikit-image (skimage)
- Pillow (PIL)

Usage:
------
python edge_cleaner.py -i <input_path> -o <output_path> [options]

Arguments:
----------
  -i, --input      Path to a single PNG file or a folder containing PNG files.
  -o, --output     Path to the folder where cleaned images will be saved.
  --threshold      Grayscale threshold for binarization (default: 200).
  --delta          Buffer added to the erasure radius to clean up antialiased edges (default: 1.5).
  --close_ksize    Kernel size for morphological closing of dashed lines (default: 9).
  --inpaint_rad    Inpainting radius for smoothing out intersection cuts (default: 3).

Example:
--------
python edge_cleaner.py -i "Client Provided Source" -o "test_output"
"""

import os
import sys
import argparse
import numpy as np
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import cv2
from PIL import Image

def clean_image_edges(image_path, threshold=200, delta=1.5, close_ksize=9, inpaint_rad=3):
    """
    Cleans the edge-connected lines of a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input PNG image.
    threshold : int
        Grayscale threshold to convert the image to binary (foreground=1, background=0).
    delta : float
        Additional buffer added to the distance transform radius to fully clean up
        gray antialiased boundaries.
    close_ksize : int
        Kernel size for morphological closing to bridge dashed lines before tracing.
    inpaint_rad : int
        Radius for the OpenCV inpainting to restore intersecting boundaries.
        
    Returns:
    --------
    cleaned_arr : numpy.ndarray
        Grayscale cleaned image.
    border_removed : int
        Number of border-crossing pixels removed.
    total_removed : int
        Total number of foreground pixels removed.
    """
    # Load image as grayscale
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    h, w = arr.shape
    
    # 1. Binarize (foreground = 1, background = 0)
    binary = (arr < threshold).astype(np.uint8)
    
    # 2. Bridge dashed lines using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Skeletonize the closed binary mask to get 1-pixel wide line centerlines
    skeleton = morphology.skeletonize(closed)
    
    # 4. Compute distance transform on original binary to get exact line radii
    dist_transform = ndimage.distance_transform_edt(binary)
    
    # 5. Classify neighbors in skeleton to locate junctions
    skeleton_int = skeleton.astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    neighbors_count = ndimage.convolve(skeleton_int, neighbor_kernel, mode='constant', cval=0)
    
    # Find border mask
    border_mask = np.zeros_like(binary, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    
    # Find border pixels that are part of the foreground
    border_pixels = np.argwhere(border_mask & (closed == 1))
    
    visited_skeleton = np.zeros_like(skeleton, dtype=bool)
    erase_mask = np.zeros_like(binary, dtype=np.uint8)
    
    for by, bx in border_pixels:
        # Determine initial direction inwards from the border
        if by == 0:
            init_dir = (1, 0)
        elif by == h - 1:
            init_dir = (-1, 0)
        elif bx == 0:
            init_dir = (0, 1)
        else:
            init_dir = (0, -1)
            
        # BFS to find the nearest skeleton pixel starting from the border
        queue = [[(by, bx)]]
        visited_bfs = np.zeros_like(binary, dtype=bool)
        visited_bfs[by, bx] = True
        
        found_skeleton_pixel = None
        bfs_path = []
        
        while queue:
            path = queue.pop(0)
            cy, cx = path[-1]
            
            if skeleton[cy, cx]:
                found_skeleton_pixel = (cy, cx)
                bfs_path = path
                break
                
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if closed[ny, nx] == 1 and not visited_bfs[ny, nx]:
                            visited_bfs[ny, nx] = True
                            queue.append(path + [(ny, nx)])
                            
        if found_skeleton_pixel:
            sy, sx = found_skeleton_pixel
            
            # If the nearest skeleton pixel was already traced, we still want to erase the BFS path
            if visited_skeleton[sy, sx]:
                for py, px in bfs_path:
                    r = dist_transform[py, px]
                    radius = int(np.ceil(r + delta))
                    cv2.circle(erase_mask, (px, py), radius, 255, -1)
                continue
                
            # Trace straight along the skeleton
            skeleton_path = []
            cy, cx = sy, sx
            dy, dx = init_dir
            visited_skeleton[cy, cx] = True
            skeleton_path.append((cy, cx))
            
            r_line = dist_transform[sy, sx]
            r_line = max(r_line, 1.0)
            
            reached_opposite_border = False
            
            while True:
                # Find unvisited skeleton neighbors
                neighbors = []
                for ndy in [-1, 0, 1]:
                    for ndx in [-1, 0, 1]:
                        if ndy == 0 and ndx == 0:
                            continue
                        ny, nx = cy + ndy, cx + ndx
                        if 0 <= ny < h and 0 <= nx < w:
                            if skeleton[ny, nx] and not visited_skeleton[ny, nx]:
                                neighbors.append((ny, nx, ndy, ndx))
                                
                if not neighbors:
                    # Check if we are at the border
                    if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                        reached_opposite_border = True
                    break
                    
                # Choose the neighbor that continues most straight
                best_neighbor = None
                best_dot = -2.0  # Cosine similarity range is [-1, 1]
                for ny, nx, ndy, ndx in neighbors:
                    norm_prev = np.sqrt(dy**2 + dx**2)
                    norm_curr = np.sqrt(ndy**2 + ndx**2)
                    dot = (dy * ndy + dx * ndx) / (norm_prev * norm_curr)
                    if dot > best_dot:
                        best_dot = dot
                        best_neighbor = (ny, nx, ndy, ndx)
                        
                # Junction crossing logic:
                # If we are at a junction, we only cross if we go almost exactly straight (dot > 0.85)
                # Otherwise, we can continue if dot > 0.4
                is_junction = (neighbors_count[cy, cx] >= 3)
                threshold_dot = 0.85 if is_junction else 0.4
                
                if best_neighbor and best_dot > threshold_dot:
                    ny, nx, ndy, ndx = best_neighbor
                    cy, cx = ny, nx
                    dy, dx = ndy, ndx  # update direction
                    visited_skeleton[cy, cx] = True
                    skeleton_path.append((cy, cx))
                    
                    # Check distance transform stopping condition (entering filled shape)
                    d = dist_transform[cy, cx]
                    if d > r_line + 3.0 or d > 1.35 * r_line:
                        break
                else:
                    break
            
            # Combine paths
            full_path = bfs_path + skeleton_path
            
            # Backtrack to avoid biting into symbols
            if reached_opposite_border:
                filtered_path = full_path
            else:
                # Backtrack by r_line using Euclidean distance
                accum_dist = 0.0
                idx = len(full_path) - 1
                while idx > 0 and accum_dist < r_line:
                    p1 = full_path[idx]
                    p2 = full_path[idx-1]
                    step_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    accum_dist += step_dist
                    idx -= 1
                filtered_path = full_path[:idx + 1]
                
            # Draw circles of radius dist_transform + delta around each path pixel to erase the line
            for py, px in filtered_path:
                r = dist_transform[py, px]
                radius = int(np.ceil(r + delta))
                cv2.circle(erase_mask, (px, py), radius, 255, -1)
                

    # 6. Detect horizontal and vertical dashed/solid lines morphologically
    # Dilate horizontally to bridge horizontal dashes, then open to keep long horizontal segments
    dilated_h = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1)))
    mask_h = cv2.morphologyEx(dilated_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)))
    num_labels_h, labeled_h = cv2.connectedComponents(mask_h)
    for i in range(1, num_labels_h):
        comp_mask = (labeled_h == i)
        if np.any(comp_mask & border_mask):
            y_indices, x_indices = np.where(comp_mask)
            comp_h = np.max(y_indices) - np.min(y_indices) + 1
            # Only erase if it is a thin horizontal line (height <= 35)
            # This protects large filled shapes (e.g. circle in symbol 1)
            if comp_h <= 35:
                comp_mask_dilated = cv2.dilate(comp_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                erase_mask[comp_mask_dilated > 0] = 255
            
    # Dilate vertically to bridge vertical dashes, then open to keep long vertical segments
    dilated_v = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
    mask_v = cv2.morphologyEx(dilated_v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)))
    num_labels_v, labeled_v = cv2.connectedComponents(mask_v)
    for i in range(1, num_labels_v):
        comp_mask = (labeled_v == i)
        if np.any(comp_mask & border_mask):
            y_indices, x_indices = np.where(comp_mask)
            comp_w = np.max(x_indices) - np.min(x_indices) + 1
            # Only erase if it is a thin vertical line (width <= 35)
            # This protects large vertical shapes (e.g. rectangle in symbol 8)
            if comp_w <= 35:
                comp_mask_dilated = cv2.dilate(comp_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                erase_mask[comp_mask_dilated > 0] = 255
            
    # Also handle separate connected components that touch the border and are small
    # (e.g. text characters 'X' or separate noise that is isolated)
    # Using relative area threshold to handle thick text characters in large images
    labeled, num_features = ndimage.label(binary, structure=np.ones((3,3)))
    area_thresh = max(3000, int(h * w * 0.015))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    for label_idx in range(1, num_features + 1):
        comp_mask = (labeled == label_idx)
        touches_border = np.any(comp_mask & border_mask)
        if touches_border:
            area = np.sum(comp_mask)
            if area < area_thresh and area < (h * w * 0.05):
                comp_mask_dilated = cv2.dilate(comp_mask.astype(np.uint8), dilate_kernel)
                erase_mask[comp_mask_dilated > 0] = 255
                
    # Create final cleaned image using inpainting to smooth the intersection boundaries
    cleaned = cv2.inpaint(arr, erase_mask, inpaint_rad, cv2.INPAINT_TELEA)
    
    # Calculate how many border pixels were removed
    orig_border_pixels = arr[border_mask]
    cleaned_border_pixels = cleaned[border_mask]
    border_removed_count = np.sum((orig_border_pixels < threshold) & (cleaned_border_pixels >= threshold))
    
    total_removed_count = np.sum((arr < threshold) & (cleaned >= threshold))
    
    return cleaned, border_removed_count, total_removed_count


def main():
    parser = argparse.ArgumentParser(description="Python Edge-Line Cleaner Script")
    parser.add_argument("-i", "--input", required=True, help="Input PNG file path or folder path")
    parser.add_argument("-o", "--output", required=True, help="Output folder path")
    parser.add_argument("--threshold", type=int, default=200, help="Grayscale threshold for binarization")
    parser.add_argument("--delta", type=float, default=1.5, help="Radius expansion delta for erasure circles")
    parser.add_argument("--close_ksize", type=int, default=9, help="Kernel size for closing dashed lines")
    parser.add_argument("--inpaint_rad", type=int, default=3, help="Radius for inpainting smooth boundaries")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # Verify input exists
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)
        
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Collect PNG files
    if os.path.isdir(input_path):
        png_files = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path)) 
                     if f.lower().endswith(".png") and f.lower() != "clip_examples.png"]
    else:
        if input_path.lower().endswith(".png"):
            png_files = [input_path]
        else:
            print("Error: Input file must be a PNG image.")
            sys.exit(1)
            
    if not png_files:
        print(f"No PNG images found in input path '{input_path}'.")
        sys.exit(0)
        
    print(f"Processing {len(png_files)} image(s)...")
    print("-" * 65)
    print(f"{'Filename':<30} | {'Border Pixels Removed':<21} | {'Total Pixels Removed'}")
    print("-" * 65)
    
    for fpath in png_files:
        fname = os.path.basename(fpath)
        try:
            cleaned, border_cnt, total_cnt = clean_image_edges(
                fpath,
                threshold=args.threshold,
                delta=args.delta,
                close_ksize=args.close_ksize,
                inpaint_rad=args.inpaint_rad
            )
            # Save final output
            out_fpath = os.path.join(output_path, fname)
            cv2.imwrite(out_fpath, cleaned)
            print(f"{fname:<30} | {border_cnt:<21} | {total_cnt}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            
    print("-" * 65)
    print("Done!")

if __name__ == "__main__":
    main()
