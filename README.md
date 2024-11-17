# Image Perspective Transformer

This project provides a PyQt-based GUI application for marking four points on an image to apply a perspective transformation. Users can preview the transformed image and reselect points if the result is unsatisfactory.

---

## Features
- Load and Display Image: Load an image and scale it to fit the screen while maintaining its aspect ratio.
- Mark Points: Click on the image to select four points for transformation.
- Clear Points: Reset the marked points and start over without restarting the application.
- Preview Transformation: Display the transformed image and check its quality before saving.
- Iterative Workflow: If the transformation is not satisfactory, reselect the points and apply the transformation again.
- Save Image: Save the transformed image to a file.
- Exit Application: Close the application. (Ctrl+Q / Command+Q)

---

## Ensure the following are installed on your system:
- Python 3.7+

## Install Steps
1. Clone the repository:
```bash
git clone https://github.com/zangjiucheng/Perspective-Transformer.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python perspective.py <image_path> [output_path]
```

--- 

## How to Use
1. Run the Application:

    - Replace image.jpg in the code with the path to your image or place your image in the same directory as the script.
    - Start the application:

        ```bash
        python perspective.py <image_path> [output_path]
        ```
2. Mark Points:

    Click on the image to select four points in any order. These points define the region for perspective transformation.

3. Confirm Points:

    After selecting four points, click "Confirm Points" to apply the transformation.

4. Preview Transformed Image:

    Click "Preview Transformed Image" to see the transformed result in the same window.
    If the result is unsatisfactory, click "Clear Points" to reset and reselect points.

5. Save Transformed Image:

    When satisfied, the application automatically saves the transformed image as transformed_image.jpg or the specified output path.

--- 

## License

This project is open-source and can be modified and used for personal or educational purposes. Attribution to the original creator is appreciated. (MIT License)

Enjoy transforming your images! 🚀