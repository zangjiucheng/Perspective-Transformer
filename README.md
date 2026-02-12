# Image Perspective Transformer

This project provides a PySide6 GUI for marking four points on an image, applying a perspective transform, and exporting a clean crop for documents (e.g., LaTeX). It supports interactive point editing, live preview, and automatic downscaling to a maximum side length for consistent output sizes.

---

## Features
- Smart image view: Fit-to-screen display with accurate coordinate mapping.
- Fast point selection: Click up to four points to define the transform region.
- Editable points: Drag points to refine corners; lines and area shading update live.
- Area highlight: The selected quadrilateral is shaded for clear visual feedback.
- Detail magnifier: Hold Ctrl near a point for a 2x zoom lens on the corner.
- Preview + export: Preview the transformed image and save to a chosen path.
- Max-side downscale: Cap output size via `--max-side` (default 1024) for LaTeX-friendly images.
- Shortcuts: Ctrl+Q to quit, Ctrl+S to save, Ctrl+P to preview, Ctrl+Enter to apply.

---

## Ensure the following are installed on your system:
- Python 3.7+

## Install Steps
1. Install from PyPI:
```bash
pip install perspective-transformer
```

2. Run the application:
```bash
perspective-transformer <image_path> [output_path] [--max-side 1024]
```

--- 

## How to Use
1. Run the Application:

    - Replace image.jpg in the code with the path to your image or place your image in the same directory as the script.
    - Start the application:

        ```bash
        perspective-transformer <image_path> [output_path] [--max-side 1024]
        ```
2. Mark Points:

    Click on the image to select four points in any order. These points define the region for perspective transformation. Drag any point to adjust.

3. Confirm Points:

    After selecting four points, click "Select Points" (or Ctrl+Enter) to apply the transformation.

4. Preview Transformed Image:

    Click "Preview Transformed Image" to see the transformed result in the same window.
    If the result is unsatisfactory, click "Clear Points" to reset and reselect points.

5. Save Transformed Image:

    When satisfied, click "Write Transformed Image" (or Ctrl+S) to save the output as transformed_image.jpg or the specified output path.

--- 

## License

This project is open-source and can be modified and used for personal or educational purposes. Attribution to the original creator is appreciated. (MIT License)

Enjoy transforming your images! 🚀
