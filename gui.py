import os
import sys
import subprocess
import time # Added for timestamping
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QSlider, QFileDialog,
                             QComboBox, QDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap # QImageReader is not used directly, QPixmap handles it.


# --- Helper Class for Clickable Labels ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path_internal = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def set_image_path(self, path):
        self.image_path_internal = path

    def get_image_path(self):
        return self.image_path_internal


class IPGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle("IP программа настройки параметров") # Set later with translations
        self.setGeometry(100, 100, 900, 700)

        # --- Translations ---
        self.translations = {
            "window_title": "Изменение атрибутов лица",
            "select_input_image_btn": "Выбрать входное изображение",
            "no_image_selected_label": "Входное изображение не выбрано",
            "direction_attr_label": "Атрибут направления:",
            "alpha_value_label_prefix": "Значение Alpha:",
            "run_program_btn": "Запустить программу",
            "exit_btn": "Выход", # New translation
            "input_image_header": "Входное изображение:",
            "output_image_header": "Результат обработки:",
            "input_image_placeholder": "Входное изображение будет здесь",
            "output_image_placeholder": "Результат будет здесь",
            "select_input_image_dialog_title": "Выбрать входное изображение",
            "status_select_image_first": "Пожалуйста, сначала выберите входное изображение",
            "status_program_started": "Программа запущена",
            "status_run_failed": "Ошибка запуска: {error}",
            "status_displaying_result": "Отображение результата: {filename}",
            "status_ready": "Готово",
            "status_direction_file_not_found": "Файл направления не найден: {filepath}",
            "large_image_view_title": "Просмотр изображения",
            "failed_to_load_image_for_viewing": "Не удалось загрузить изображение для просмотра.",
            "failed_to_load_thumbnail": "Не удалось загрузить: {filename}"
        }
        self.setWindowTitle(self.translations["window_title"])


        # Initialize variables
        self.image_path = ""
        self.direction_attr = ""
        self.alpha = 20
        self.output_dir = "output"
        self.process = None # To store the subprocess

        # For tracking output file updates
        self.last_run_start_time = 0
        self.last_displayed_output_mtime = -1 # Ensures first valid output of a run is shown
        self.last_displayed_output_path = None


        self.direction2idx = {
            'Bushy_Eyebrows': 6, 'Eyeglasses': 7, 'Mouth_Open': 10,
            'Narrow_Eyes': 11, 'Beard': 12, 'Smiling': 15, 'Old': 16
        }

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.image_label_info = QLabel(self.translations["no_image_selected_label"])
        image_button = QPushButton(self.translations["select_input_image_btn"])
        image_button.clicked.connect(self.select_image_file)

        direction_layout = QHBoxLayout()
        direction_label = QLabel(self.translations["direction_attr_label"])
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(self.direction2idx.keys())
        self.direction_combo.currentTextChanged.connect(self.update_direction_attr)
        direction_layout.addWidget(direction_label)
        direction_layout.addWidget(self.direction_combo)

        alpha_layout = QHBoxLayout()
        alpha_label_text = QLabel(self.translations["alpha_value_label_prefix"])
        self.alpha_value_label = QLabel(str(self.alpha))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 50)
        self.alpha_slider.setValue(self.alpha)
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        alpha_layout.addWidget(alpha_label_text)
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_value_label)

        # --- Run and Exit Buttons ---
        buttons_layout = QHBoxLayout()
        run_button = QPushButton(self.translations["run_program_btn"])
        run_button.clicked.connect(self.run_program)
        exit_button = QPushButton(self.translations["exit_btn"])
        exit_button.clicked.connect(self.close) # QMainWindow.close() handles app exit if it's the main window
        buttons_layout.addWidget(run_button)
        buttons_layout.addWidget(exit_button)


        image_display_layout = QHBoxLayout()
        input_image_group_layout = QVBoxLayout()
        input_image_header_label = QLabel(self.translations["input_image_header"])
        input_image_header_label.setAlignment(Qt.AlignCenter)
        self.input_image_label_display = ClickableLabel(self.translations["input_image_placeholder"])
        self.input_image_label_display.setAlignment(Qt.AlignCenter)
        self.input_image_label_display.setStyleSheet("border: 1px solid black;")
        self.input_image_label_display.setMinimumSize(300, 300)
        self.input_image_label_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_image_label_display.clicked.connect(
            lambda: self.show_large_image(self.input_image_label_display.get_image_path())
        )
        input_image_group_layout.addWidget(input_image_header_label)
        input_image_group_layout.addWidget(self.input_image_label_display)

        output_image_group_layout = QVBoxLayout()
        output_image_header_label = QLabel(self.translations["output_image_header"])
        output_image_header_label.setAlignment(Qt.AlignCenter)
        self.output_image_label_display = ClickableLabel(self.translations["output_image_placeholder"])
        self.output_image_label_display.setAlignment(Qt.AlignCenter)
        self.output_image_label_display.setStyleSheet("border: 1px solid black;")
        self.output_image_label_display.setMinimumSize(300, 300)
        self.output_image_label_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_image_label_display.clicked.connect(
            lambda: self.show_large_image(self.output_image_label_display.get_image_path())
        )
        output_image_group_layout.addWidget(output_image_header_label)
        output_image_group_layout.addWidget(self.output_image_label_display)

        image_display_layout.addLayout(input_image_group_layout)
        image_display_layout.addLayout(output_image_group_layout)

        layout.addWidget(QLabel(self.translations["input_image_header"]))
        layout.addWidget(image_button)
        layout.addWidget(self.image_label_info)
        layout.addLayout(direction_layout)
        layout.addLayout(alpha_layout)
        layout.addLayout(buttons_layout) # Add the QHBoxLayout for buttons
        layout.addLayout(image_display_layout)

        self.statusBar().showMessage(self.translations["status_ready"])

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_output_folder)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.update_direction_attr(self.direction_combo.currentText())

    def select_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.translations["select_input_image_dialog_title"], "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.image_path = file_path
            self.image_label_info.setText(os.path.basename(file_path))
            self.display_image_thumbnail(file_path, self.input_image_label_display)

    def update_direction_attr(self, attr):
        self.direction_attr = attr
        self.direction_path = f"./directions/{attr}.npy"

    def update_alpha(self, value):
        self.alpha = value
        self.alpha_value_label.setText(str(value))

    def display_image_thumbnail(self, image_path, label_widget: ClickableLabel):
        if not image_path or not os.path.exists(image_path):
            label_widget.setText(self.translations["input_image_placeholder"])
            label_widget.set_image_path(None)
            return

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            label_widget.set_image_path(image_path)
            scaled_pixmap = pixmap.scaled(
                label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label_widget.setPixmap(scaled_pixmap)
        else:
            label_widget.setText(self.translations["failed_to_load_thumbnail"].format(filename=os.path.basename(image_path)))
            label_widget.set_image_path(None)

    def show_large_image(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.translations['large_image_view_title']} - {os.path.basename(image_path)}")
        layout_ = QVBoxLayout(dialog) # Use layout_ to avoid conflict with main layout
        image_label = QLabel(dialog)
        image_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            image_label.setText(self.translations["failed_to_load_image_for_viewing"])
        else:
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            max_width = screen_geometry.width() * 0.8
            max_height = screen_geometry.height() * 0.8
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(int(max_width), int(max_height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)

        layout_.addWidget(image_label)
        dialog.setLayout(layout_)
        dialog.adjustSize()
        dialog.exec_()

    def run_program(self):
        if not self.image_path:
            self.statusBar().showMessage(self.translations["status_select_image_first"], 3000)
            return
        if not os.path.exists(self.direction_path):
            self.statusBar().showMessage(
                self.translations["status_direction_file_not_found"].format(filepath=self.direction_path), 5000)
            return

        self.last_run_start_time = time.time()
        # Reset last displayed state for this specific run
        self.last_displayed_output_mtime = self.last_run_start_time -1 # Allow first file to be displayed
        self.last_displayed_output_path = None


        # Clear previous output image display
        self.output_image_label_display.setText(self.translations["output_image_placeholder"])
        self.output_image_label_display.set_image_path(None)
        self.output_image_label_display.repaint() # Force immediate repaint
        QApplication.processEvents()


        command = [
            sys.executable, "forgui.py",
            "--image_path", self.image_path,
            "--direction_path", self.direction_path,
            "--alpha", str(self.alpha),
            "--output_dir", self.output_dir
        ]
        print(f"Running command: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(command)
            self.statusBar().showMessage(self.translations["status_program_started"], 2000)
            if not self.timer.isActive():
                self.timer.start(1000) # Check every 1 second
        except Exception as e:
            self.statusBar().showMessage(self.translations["status_run_failed"].format(error=str(e)), 5000)
            self.process = None # Ensure process is None if Popen failed


    def _find_latest_output_file(self):
        """Finds the most recently modified image file in output_dir from the current run."""
        if not os.path.exists(self.output_dir) or self.last_run_start_time == 0:
            return None

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        candidate_files_info = []

        for f_name in os.listdir(self.output_dir):
            if f_name.lower().endswith(image_extensions):
                f_path = os.path.join(self.output_dir, f_name)
                try:
                    mtime = os.path.getmtime(f_path)
                    # Check if file was modified during or after the current run started
                    if mtime >= self.last_run_start_time:
                        candidate_files_info.append({'path': f_path, 'name': f_name, 'mtime': mtime})
                except FileNotFoundError:
                    continue # File might have been deleted

        if candidate_files_info:
            candidate_files_info.sort(key=lambda item: item['mtime'], reverse=True)
            return candidate_files_info[0] # Returns dict: {'path': ..., 'name': ..., 'mtime': ...}
        return None

    def check_output_folder(self):
        if self.process is None and not self.timer.isActive(): # No active process, timer shouldn't run
            return
        if self.process is None and self.timer.isActive(): # Process somehow became None but timer is running
            self.timer.stop()
            return


        process_finished = self.process is not None and self.process.poll() is not None

        latest_output_file_info = self._find_latest_output_file()

        if latest_output_file_info:
            path = latest_output_file_info['path']
            name = latest_output_file_info['name']
            mtime = latest_output_file_info['mtime']

            # Display if it's genuinely newer than what was last shown for this run
            if mtime > self.last_displayed_output_mtime or \
               (mtime == self.last_displayed_output_mtime and path != self.last_displayed_output_path):
                # Using singleShot to give file system a moment and avoid GUI freeze on load
                QTimer.singleShot(200, lambda p=path, n=name, mt=mtime: self.finalize_output_display(p, n, mt))

        if process_finished:
            self.timer.stop()
            print(f"External process finished with code: {self.process.returncode if self.process else 'N/A'}.")
            self.process = None # Clear the process
            # If a final output was written just as process ended, the singleShot above should catch it.

    def finalize_output_display(self, image_path, image_name, image_mtime):
        # This method is called by QTimer.singleShot
        # Check if file still exists, as 200ms have passed
        if not os.path.exists(image_path):
            print(f"Output file {image_name} disappeared before display.")
            return

        print(f"Displaying output: {image_path} (mtime: {image_mtime})")
        self.display_image_thumbnail(image_path, self.output_image_label_display)
        self.statusBar().showMessage(
            self.translations["status_displaying_result"].format(filename=image_name), 3000
        )
        self.last_displayed_output_path = image_path
        self.last_displayed_output_mtime = image_mtime


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.input_image_label_display.get_image_path():
            self.display_image_thumbnail(self.input_image_label_display.get_image_path(), self.input_image_label_display)
        if self.output_image_label_display.get_image_path():
            self.display_image_thumbnail(self.output_image_label_display.get_image_path(), self.output_image_label_display)

    def closeEvent(self, event):
        # Clean up: terminate subprocess if it's running
        if self.process and self.process.poll() is None:
            print("Terminating active subprocess...")
            self.process.terminate()
            try:
                self.process.wait(timeout=1) # Wait a bit for it to terminate
            except subprocess.TimeoutExpired:
                print("Subprocess did not terminate in time, killing.")
                self.process.kill()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IPGUI()
    window.show()
    sys.exit(app.exec_())