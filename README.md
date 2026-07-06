# Python Edge-Line Cleaner Script

A command-line Python utility that scans PNG schematics, detects all black-colored pixels (connection leads/wires) connected to the outer boundary frame, and erases them while keeping interior elements (text, circles, boxes, switches, fine lines) fully intact.

---

## Features
- **BFS-guided Skeletonization**: Bridges the gap between boundary pixels and centerline skeletons, ensuring lines are traced all the way to the boundary.
- **Euclidean Backtracking**: Measures path distances to stop erasure precisely at symbol boundaries, preventing circular "bites" or artifacts on interior marks.
- **Junction Constraint Routing**: Analyzes cosine similarity of neighbor paths at junctions to trace straight crossing lines (like vertical wires passing through circles) while stopping at T/Y junctions (where leads join symbol bodies).
- **Automated Text Cleanup**: Detects and completely erases isolated border-touching text components (e.g., "X X") while dilating to clear antialiased edges.
- **Seamless Inpainting**: Employs OpenCV inpainting to restore intersecting boundaries, making circle rings and symbols perfectly continuous.

---

## Installation

Ensure you have Python 3.8+ installed. Install the required image processing libraries:

```bash
pip install numpy opencv-python scipy scikit-image pillow
```

---

## Usage

You can point the script at a single PNG file or a folder of PNG files.

```bash
python edge_cleaner.py -i <input_path> -o <output_path> [options]
```

### Arguments:
*   `-i`, `--input` (Required): Path to a single PNG image or folder of PNG images.
*   `-o`, `--output` (Required): Path to the output directory where cleaned images will be saved.
*   `--threshold` (Optional, default `200`): Grayscale threshold for binarizing drawings (0 to 255).
*   `--delta` (Optional, default `1.5`): Expansion buffer added to the line width to clear antialiased gray edges.
*   `--close_ksize` (Optional, default `9`): Kernel size for morphological closing of dashed lines.
*   `--inpaint_rad` (Optional, default `3`): Inpainting radius for boundary reconstruction.

### Examples:

1.  **Clean all images in the client provided folder:**
    ```bash
    python edge_cleaner.py -i "Client Provided Source" -o "test_output"
    ```
2.  **Clean a single file with custom binarization threshold:**
    ```bash
    python edge_cleaner.py -i "Client Provided Source/page_001_symbol_001.png" -o "output" --threshold 180
    ```
