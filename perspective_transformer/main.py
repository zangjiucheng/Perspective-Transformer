import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QGridLayout
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence
from PySide6.QtCore import Qt
import cv2
import numpy as np


class ImageMarker(QMainWindow):
    def __init__(self, image_path,output_path):
        super().__init__()
        self.image_path = image_path
        self.output_path = output_path
        self.points = []  # Store clicked points
        self.scale_factor_x = 1  # X-axis scaling factor
        self.scale_factor_y = 1  # Y-axis scaling factor

        self.transformed_image = None  # Store the transformed image

        self.initUI()


    def initUI(self):
        # Load the image using OpenCV
        self.cv_image = cv2.imread(self.image_path)
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.original_image = self.cv_image.copy()
        height, width, channel = self.cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert to QPixmap
        self.pixmap = QPixmap.fromImage(q_image)

        # Scale image to fit the screen
        screen_rect = QApplication.primaryScreen().geometry()
        screen_width, screen_height = screen_rect.width(), screen_rect.height()
        self.scaled_pixmap = self.pixmap.scaled(
            screen_width * 0.8, screen_height * 0.8, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Calculate scale factors
        self.scale_factor_x = self.pixmap.width() / self.scaled_pixmap.width()
        self.scale_factor_y = self.pixmap.height() / self.scaled_pixmap.height()

        # Create QLabel to display image
        self.label = QLabel()
        self.label.setPixmap(self.scaled_pixmap)
        self.label.mousePressEvent = self.mouse_click  # Bind mouse click event

        # Add buttons
        self.confirm_button = QPushButton("Select Points")
        self.confirm_button.clicked.connect(self.select_points)

        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)

        self.preview_button = QPushButton("Preview Transformed Image")
        self.preview_button.clicked.connect(self.preview_transformed_image)

        self.write_button = QPushButton("Write Transformed Image")
        self.write_button.clicked.connect(self.write_transformed_image)

        # Add key binding to close the program
        self.shortcut_close = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.shortcut_close.activated.connect(self.close)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.write_transformed_image)
        
        self.shortcut_preview = QShortcut(QKeySequence("Ctrl+P"), self)
        self.shortcut_preview.activated.connect(self.preview_transformed_image)
        
        self.shortcut_select = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_select.activated.connect(self.select_points)

        # Layout
        layout = QVBoxLayout()
        button_layout = QGridLayout()
        button_layout.addWidget(self.confirm_button, 0, 0)
        button_layout.addWidget(self.clear_button, 0, 1)
        button_layout.addWidget(self.preview_button, 1, 0)
        button_layout.addWidget(self.write_button, 1, 1)

        layout.addWidget(self.label)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set window properties
        self.setWindowTitle("Image Marker")
        self.resize(self.scaled_pixmap.width(), self.scaled_pixmap.height() + 50)

    def mouse_click(self, event):
        # Get the click position on the scaled image
        if event.button() == Qt.LeftButton and len(self.points) < 4:
            x = event.pos().x() * self.scale_factor_x
            y = event.pos().y() * self.scale_factor_y
            self.points.append((int(x), int(y)))
            print(f"Point {len(self.points)}: ({int(x)}, {int(y)})")
            
            # Draw the point on the original image
            self.cv_image = cv2.circle(self.cv_image, (int(x), int(y)), 15, (255, 0, 0), -1)
            height, width, channel = self.cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.scaled_pixmap.width(), self.scaled_pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            if len(self.points) == 4:
                print("All points selected. Click 'Select Points' to proceed.")
                self.connect_points()

    def select_points(self):
        if len(self.points) == 4:
            print("Selected Points:", self.points)
            self.apply_perspective_transform()
        else:
            print("Please select exactly 4 points.")

    def connect_points(self):
        # Connect the points with lines
        p1, p2, p3, p4 = self.order_points(self.points)
        self.cv_image = cv2.line(self.cv_image, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 5)
        self.cv_image = cv2.line(self.cv_image, tuple(map(int, p2)), tuple(map(int, p3)), (0, 255, 0), 5)
        self.cv_image = cv2.line(self.cv_image, tuple(map(int, p3)), tuple(map(int, p4)), (0, 255, 0), 5)
        self.cv_image = cv2.line(self.cv_image, tuple(map(int, p4)), tuple(map(int, p1)), (0, 255, 0), 5)

        # Update QLabel with the connected points
        height, width, channel = self.cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.scaled_pixmap.width(), self.scaled_pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    
    def clear_points(self):
        # Reset points and reload the original image
        self.points = []
        print("Cleared all points.")
        self.cv_image = self.original_image.copy()

        # Reset QLabel with the original image
        height, width, channel = self.cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.scaled_pixmap.width(), self.scaled_pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))


    def calculate_dimensions(self):
        # Calculate the width and height of the quadrilateral
        p1, p2, p3, p4 = self.order_points(self.points)

        width_top = np.linalg.norm(np.array(p1) - np.array(p2))
        width_bottom = np.linalg.norm(np.array(p3) - np.array(p4))
        height_left = np.linalg.norm(np.array(p1) - np.array(p4))
        height_right = np.linalg.norm(np.array(p2) - np.array(p3))

        width = int(max(width_top, width_bottom))
        height = int(max(height_left, height_right))

        return width, height

    def order_points(self, points):
        # Convert to numpy array for easier manipulation
        points = np.array(points, dtype="float32")

        # Order points by y-coordinates (top to bottom)
        sorted_points = points[np.argsort(points[:, 1])]

        # Extract top two and bottom two points
        top_points = sorted_points[:2]
        bottom_points = sorted_points[2:]

        # Sort top points by x-coordinate to get top-left and top-right
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]

        # Sort bottom points by x-coordinate to get bottom-left and bottom-right
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

        # Return ordered points: [top-left, top-right, bottom-right, bottom-left]
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    def apply_perspective_transform(self):
        # Define the target rectangle's size
        width, height = self.calculate_dimensions()
        
        dst_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # Compute the perspective transform matrix
        src_points = self.order_points(np.float32(self.points))
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation
        self.transformed_image = cv2.warpPerspective(self.original_image, matrix, (width, height))
        print("Transformation applied. Preview the result.")


    def write_transformed_image(self):
        if self.transformed_image is not None:
            # Save the transformed image
            cv2.imwrite(self.output_path, cv2.cvtColor(self.transformed_image, cv2.COLOR_RGB2BGR))
            print(f"Transformed image saved as {self.output_path}")
        else:
            print("No transformation has been applied yet)")
    
    def preview_transformed_image(self):
        if self.transformed_image is not None:
            # Display the transformed image in the QLabel
            height, width, channel = self.transformed_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.transformed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.scaled_pixmap.width(), self.scaled_pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            print("Previewing transformed image. Click 'Clear Points' to reselect if unsatisfactory.")
        else:
            print("No transformation has been applied yet.")

def main():
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python perspective.py <image_path> [output_path]")
        sys.exit(1)
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else "transformed_image.jpg"

    window = ImageMarker(image_path,output_path)
    window.show()
    sys.exit(app.exec_())

# Run the application
if __name__ == "__main__":
    main()