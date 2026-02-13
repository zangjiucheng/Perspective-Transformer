import sys
import argparse
from pathlib import Path
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QGridLayout
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QIcon
from PySide6.QtCore import Qt
import cv2
import numpy as np


class ImageMarker(QMainWindow):
    def __init__(self, image_path,output_path):
        super().__init__()
        self.image_path = image_path
        self.output_path = output_path
        self.max_side = 1024
        self.points = []  # Store clicked points
        self.drag_index = None
        self.magnify_point = None
        self.magnify_scale = 2.0
        self.magnify_ratio = 0.15
        self.scale_factor_x = 1  # X-axis scaling factor
        self.scale_factor_y = 1  # Y-axis scaling factor

        self.transformed_image = None  # Store the transformed image

        self.initUI()


    def initUI(self):
        # Load the image using OpenCV
        self.cv_image = cv2.imread(self.image_path)
        if self.cv_image is None:
            raise FileNotFoundError(
                f"Unable to read image at '{self.image_path}'. "
                "Check the path and file format."
            )
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
        self.label.setMouseTracking(True)
        self.label.setCursor(Qt.ArrowCursor)
        self.label.mousePressEvent = self.mouse_press  # Bind mouse click event
        self.label.mouseMoveEvent = self.mouse_move
        self.label.mouseReleaseEvent = self.mouse_release

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

        self.shortcut_clear = QShortcut(QKeySequence("Ctrl+L"), self)
        self.shortcut_clear.activated.connect(self.clear_points)

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

    def mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return

        click_point = self._event_to_image_point(event)
        drag_index = self._find_near_point(click_point)
        if drag_index is not None:
            self.drag_index = drag_index
            return

        if len(self.points) < 4:
            self.points.append(click_point)
            print(f"Point {len(self.points)}: ({click_point[0]}, {click_point[1]})")
            self.redraw_points()
            if len(self.points) == 4:
                print("All points selected. Click 'Select Points' to proceed.")

    def mouse_move(self, event):
        if self.drag_index is None:
            hover_point = self._event_to_image_point(event)
            if self._find_near_point(hover_point) is not None:
                self.label.setCursor(Qt.OpenHandCursor)
            else:
                self.label.setCursor(Qt.ArrowCursor)
            if event.modifiers() & Qt.ControlModifier and self._find_near_point(hover_point) is not None:
                self.magnify_point = hover_point
                self.redraw_points()
            elif self.magnify_point is not None:
                self.magnify_point = None
                self.redraw_points()
            return
        moved_point = self._event_to_image_point(event)
        self.points[self.drag_index] = moved_point
        self.label.setCursor(Qt.ClosedHandCursor)
        if event.modifiers() & Qt.ControlModifier:
            self.magnify_point = moved_point
        elif self.magnify_point is not None:
            self.magnify_point = None
        self.redraw_points()

    def mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_index = None
            self.magnify_point = None
            self.label.setCursor(Qt.ArrowCursor)
            self.redraw_points()

    def _event_to_image_point(self, event):
        pos = event.position()
        x = int(pos.x() * self.scale_factor_x)
        y = int(pos.y() * self.scale_factor_y)
        return (x, y)

    def _find_near_point(self, point, radius_screen=20):
        if not self.points:
            return None
        radius = int(radius_screen * (self.scale_factor_x + self.scale_factor_y) / 2)
        radius_sq = radius * radius
        for idx, existing in enumerate(self.points):
            dx = existing[0] - point[0]
            dy = existing[1] - point[1]
            if dx * dx + dy * dy <= radius_sq:
                return idx
        return None

    def redraw_points(self):
        # Redraw from original so lines stay clean
        self.cv_image = self.original_image.copy()
        for idx, point in enumerate(self.points):
            self.cv_image = cv2.circle(self.cv_image, point, 15, (255, 0, 0), -1)
            if idx > 0:
                prev_point = self.points[idx - 1]
                self.cv_image = cv2.line(self.cv_image, prev_point, point, (0, 255, 0), 5)
        if len(self.points) == 4:
            self.cv_image = cv2.line(
                self.cv_image, self.points[-1], self.points[0], (0, 255, 0), 5
            )
        if len(self.points) == 4:
            overlay = self.cv_image.copy()
            ordered = self.order_points(self.points)
            cv2.fillPoly(overlay, [ordered.astype(int)], (0, 255, 0))
            self.cv_image = cv2.addWeighted(overlay, 0.25, self.cv_image, 0.75, 0)
        if self.magnify_point is not None:
            self._draw_magnifier()
        height, width, channel = self.cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.scaled_pixmap.width(), self.scaled_pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _draw_magnifier(self):
        x, y = self.magnify_point
        img_h, img_w = self.cv_image.shape[:2]
        min_dim_screen = min(self.scaled_pixmap.width(), self.scaled_pixmap.height())
        radius_screen = int(min_dim_screen * self.magnify_ratio / 2)
        radius_screen = max(40, min(radius_screen, 200))
        avg_scale = (self.scale_factor_x + self.scale_factor_y) / 2
        radius_img = max(8, int(radius_screen * avg_scale))
        src_radius = max(4, int(radius_img / self.magnify_scale))

        center = (float(x), float(y))
        src_size = (int(2 * src_radius), int(2 * src_radius))
        src_patch = cv2.getRectSubPix(self.cv_image, src_size, center)
        magnified = cv2.resize(
            src_patch, (int(2 * radius_img), int(2 * radius_img)), interpolation=cv2.INTER_LINEAR
        )

        x0 = int(x - radius_img)
        y0 = int(y - radius_img)
        x1 = x0 + magnified.shape[1]
        y1 = y0 + magnified.shape[0]

        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(img_w, x1)
        dst_y1 = min(img_h, y1)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return

        patch = magnified[src_y0:src_y1, src_x0:src_x1]
        mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
        cv2.circle(
            mask,
            (patch.shape[1] // 2, patch.shape[0] // 2),
            min(patch.shape[1], patch.shape[0]) // 2,
            255,
            -1,
        )
        roi = self.cv_image[dst_y0:dst_y1, dst_x0:dst_x1]
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(patch, patch, mask=mask)
        self.cv_image[dst_y0:dst_y1, dst_x0:dst_x1] = cv2.add(bg, fg)
        cv2.circle(self.cv_image, (x, y), radius_img, (255, 255, 255), 2)

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
        self.magnify_point = None
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
        max_dim = max(width, height)
        if self.max_side and max_dim > self.max_side:
            scale = self.max_side / max_dim
            new_size = (int(width * scale), int(height * scale))
            self.transformed_image = cv2.resize(
                self.transformed_image, new_size, interpolation=cv2.INTER_AREA
            )
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
    parser = argparse.ArgumentParser(description="Perspective transformer")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="transformed_image.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Max side length in pixels; only downscale if larger (default: 1024)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parent / "assets" / "icon.png"
    if not icon_path.is_file():
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        window = ImageMarker(args.image_path, args.output_path)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)
    window.max_side = args.max_side
    window.show()
    sys.exit(app.exec())

# Run the application
if __name__ == "__main__":
    main()
