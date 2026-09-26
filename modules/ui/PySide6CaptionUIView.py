from modules.ui.BaseCaptionUIView import BaseCaptionUIView
from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.PySide6GenerateCaptionsWindowView import PySide6GenerateCaptionsWindowView
from modules.ui.PySide6GenerateMasksWindowView import PySide6GenerateMasksWindowView
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)

_PAD = pyside6_components.PAD


class _MaskCanvas(QWidget):
    """Displays the current image and forwards mouse input for mask painting.

    The pixmap is drawn centered; positions passed to the controller are relative
    to the pixmap's top-left corner, i.e. in display image pixels.
    """

    def __init__(self, parent, view: "PySide6CaptionUIView", size: int):
        super().__init__(parent)
        self.view = view
        self._pixmap = QPixmap()
        self._cursor_pos: QPoint | None = None
        self.setFixedSize(size, size)
        self.setMouseTracking(True)

    def set_image(self, pil_image: Image.Image):
        self._pixmap = QPixmap.fromImage(ImageQt(pil_image.convert("RGBA")))
        self.update()

    def _offset(self) -> QPoint:
        return QPoint(
            (self.width() - self._pixmap.width()) // 2,
            (self.height() - self._pixmap.height()) // 2,
        )

    def _image_pos(self, event) -> tuple[float, float]:
        pos = event.position().toPoint() - self._offset()
        return pos.x(), pos.y()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if not self._pixmap.isNull():
            painter.drawPixmap(self._offset(), self._pixmap)

            # brush outline
            if self._cursor_pos is not None and self.view.is_mask_editing_enabled() \
                    and self.view.controller.mask_editing_mode == 'draw':
                radius = self.view.controller.mask_draw_radius * max(self._pixmap.width(), self._pixmap.height())
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(QColor(255, 0, 0), 1))
                painter.drawEllipse(QPointF(self._cursor_pos), radius, radius)
        painter.end()

    def mousePressEvent(self, event):
        if not self.view.is_mask_editing_enabled():
            return
        # start a new stroke at the click position instead of joining the previous one
        x, y = self._image_pos(event)
        self.view.controller.mask_draw_x = x
        self.view.controller.mask_draw_y = y
        self._forward(event)

    def mouseMoveEvent(self, event):
        self._cursor_pos = event.position().toPoint()
        if self.view.is_mask_editing_enabled():
            self._forward(event)
        self.update()

    def leaveEvent(self, event):
        self._cursor_pos = None
        self.update()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps:
            self.view.controller.update_mask_draw_radius(steps)
            self.update()
        event.accept()

    def _forward(self, event):
        x, y = self._image_pos(event)
        buttons = event.buttons()
        if self.view.controller.mask_editing_mode == 'fill' \
                and not (0 <= x < self._pixmap.width() and 0 <= y < self._pixmap.height()):
            return  # flood fill needs a seed point inside the image
        self.view.controller.handle_edit_mask(
            x, y,
            bool(buttons & Qt.LeftButton),
            bool(buttons & Qt.RightButton),
            self.view.brush_alpha(),
        )


class _ArrowKeyFilter(QObject):
    """Up/Down on the text fields switch to the previous/next image."""

    def __init__(self, controller: CaptionUIController):
        super().__init__()
        self.controller = controller

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Up:
                self.controller.previous_image()
                return True
            if event.key() == Qt.Key_Down:
                self.controller.next_image()
                return True
        return False


class PySide6CaptionUIView(BaseCaptionUIView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: CaptionUIController):
        QDialog.__init__(self, parent)
        BaseCaptionUIView.__init__(self, pyside6_components)

        self.controller = controller
        controller.view = self
        self.config_ui_state = PySide6UIState(controller.config_ui_data)
        self._switching = False

        self.setWindowTitle("Dataset Tool")
        self.resize(1280, 980)
        self.finished.connect(lambda _: self.controller._release_models())

        outer = QGridLayout(self)
        outer.setContentsMargins(_PAD, _PAD, _PAD, _PAD)
        outer.setRowStretch(1, 1)
        outer.setColumnStretch(1, 1)

        top_frame = QWidget(self)
        self.build_top_bar(top_frame, controller, self.config_ui_state)
        outer.addWidget(top_frame, 0, 0, 1, 2)

        self.file_list = QListWidget(self)
        self.file_list.setFixedWidth(300)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        outer.addWidget(self.file_list, 1, 0)

        outer.addWidget(self._build_content_column(), 1, 1)

        self._key_filter = _ArrowKeyFilter(controller)
        for widget in (self.prompt_entry, self.alpha_entry):
            widget.installEventFilter(self._key_filter)
            widget.returnPressed.connect(self.save)

        for keys, fn in (("Ctrl+M", self.toggle_mask),
                         ("Ctrl+D", self.draw_mask_editing_mode),
                         ("Ctrl+F", self.fill_mask_editing_mode)):
            QShortcut(QKeySequence(keys), self, fn)

        # Enter in the text fields must not trigger a dialog default button
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

        self.controller.load_directory(include_subdirectories=controller.config_ui_data["include_subdirectories"])

    def _build_content_column(self):
        frame = QWidget(self)
        lo = pyside6_components._layout(frame)

        self.build_mask_buttons(frame)

        self.enable_mask_editing = QCheckBox("Enable Mask Editing", frame)
        self.enable_mask_editing.toggled.connect(lambda _: self.canvas.update())
        lo.addWidget(self.enable_mask_editing, 0, 2)

        self.alpha_entry = QLineEdit("1.0", frame)
        self.alpha_entry.setFixedWidth(50)
        lo.addWidget(self.alpha_entry, 0, 3)
        lo.addWidget(QLabel("Brush Alpha", frame), 0, 4)
        lo.setColumnStretch(5, 1)

        self.canvas = _MaskCanvas(frame, self, self.controller.image_size)
        lo.addWidget(self.canvas, 1, 0, 1, 6, Qt.AlignLeft | Qt.AlignTop)
        lo.setRowStretch(1, 1)

        self.prompt_entry = QLineEdit(frame)
        lo.addWidget(self.prompt_entry, 2, 0, 1, 6)

        return frame

    # --- helpers used by the canvas ---

    def is_mask_editing_enabled(self) -> bool:
        return self.enable_mask_editing.isChecked()

    def brush_alpha(self) -> float:
        try:
            return float(self.alpha_entry.text())
        except ValueError:
            return 1.0

    def _on_file_selected(self, row):
        if self._switching or row < 0:
            return
        self.controller.switch_image(row)
        self.focus_prompt()

    # --- methods called by the controller ---

    def refresh_file_list(self):
        self._switching = True
        self.file_list.clear()
        self.file_list.addItems(self.controller.image_rel_paths)
        self._switching = False

    def focus_prompt(self):
        self.prompt_entry.setFocus()

    def on_image_switched(self, old_index, new_index, prompt):
        self._switching = True
        self.file_list.setCurrentRow(new_index)
        self._switching = False
        self.refresh_image()
        self.prompt_entry.setText(prompt)

    def on_image_cleared(self):
        self.canvas.set_image(Image.new("RGB", (512, 512), (0, 0, 0)))
        self.prompt_entry.clear()

    def refresh_image(self):
        pil_image, _ = self.controller.get_display_image()
        self.canvas.set_image(pil_image)

    # --- BaseCaptionUIView implementation ---

    def stretch_column(self, frame, column):
        pyside6_components._layout(frame).setColumnStretch(column, 1)

    def save(self):
        self.controller.save(self.prompt_entry.text())

    def draw_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode('draw')
        self.canvas.update()

    def fill_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode('fill')
        self.canvas.update()

    def toggle_mask(self):
        self.controller.toggle_mask()
        self.refresh_image()

    def open_directory(self):
        new_dir = QFileDialog.getExistingDirectory(self, "", self.controller.dir or "")
        if new_dir:
            self.controller.dir = new_dir
            self.controller.load_directory(include_subdirectories=self.controller.config_ui_data["include_subdirectories"])

    def open_mask_window(self):
        self.controller.open_mask_window(self, PySide6GenerateMasksWindowView).exec()
        self.controller.switch_image(self.controller.current_image_index)

    def open_caption_window(self):
        self.controller.open_caption_window(self, PySide6GenerateCaptionsWindowView).exec()
        self.controller.switch_image(self.controller.current_image_index)

    def open_in_explorer(self):
        self.controller.open_in_explorer()
