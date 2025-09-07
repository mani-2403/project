from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt
import sys

class ClickableLabel(QLabel):
    def __init__(self, parent=None, switch_name=""):
        super().__init__(parent)
        self.switch_name = switch_name
        self.clicked = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.clicked:
            self.clicked(self.switch_name)

class NotXorGateGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NOT + XOR Gate Simulation")
        self.setGeometry(100, 100, 1000, 500)
        self.setMouseTracking(True)

        self.coord_label = QLabel(self)
        self.coord_label.setGeometry(800, 20, 180, 30)
        self.coord_label.setStyleSheet("font-size: 14px; color: blue;")

        self.switch_states = {f"switch{i+1}": False for i in range(3)}

        self.images = {
            "battery": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\battery.jpeg"),
            "switch_on": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\switch_on.png"),
            "switch_off": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\switch_off.png"),
            "ic_not": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\NOT_gate.jpeg"),
            "ic_xor": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\EXOR_gate.jpeg"),
            "led_on": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\led_on.png"),
            "led_off": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\led_off.png"),
            "ground": QPixmap(r"C:\Users\mani8\OneDrive\Pictures\Screenshots\component\GND.jpeg"),
        }

        self.labels = {}

        self.place_image("battery", 50, 30, 60, 60)
        self.place_image("ground", 50, 400, 60, 60)

        self.place_clickable_switch("switch1", 50, 120, 60, 60)  # A
        self.place_clickable_switch("switch2", 50, 220, 60, 60)  # B
        self.place_clickable_switch("switch3", 50, 320, 60, 60)  # C

        self.place_image("ic_not", 450, 150, 160, 120)
        self.place_image("ic_xor", 250, 200, 160, 120)
        self.place_image("led_output", 700, 180, 60, 80)

        self.update_images()

    def place_image(self, name, x, y, width=60, height=60):
        lbl = QLabel(self)
        lbl.setGeometry(x, y, width, height)
        lbl.setScaledContents(True)
        lbl.setMouseTracking(True)
        self.labels[name] = lbl

    def place_clickable_switch(self, name, x, y, width=60, height=60):
        lbl = ClickableLabel(self, switch_name=name)
        lbl.setGeometry(x, y, width, height)
        lbl.setScaledContents(True)
        lbl.clicked = self.toggle_switch
        self.labels[name] = lbl

    def toggle_switch(self, name):
        self.switch_states[name] = not self.switch_states[name]
        self.update_images()
        self.update()

    def update_images(self):
        for name, label in self.labels.items():
            w, h = label.width(), label.height()

            if name.startswith("switch"):
                pixmap = self.images["switch_on"] if self.switch_states[name] else self.images["switch_off"]

            elif name == "led_output":
                A = self.switch_states["switch1"]
                B = self.switch_states["switch2"]
                C = self.switch_states["switch3"]
                not_B = not B
                output = (A ^ not_B) ^ C
                pixmap = self.images["led_on"] if output else self.images["led_off"]

            else:
                pixmap = self.images[name]

            label.setPixmap(pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio))

    def mouseMoveEvent(self, event):
        x, y = event.position().x(), event.position().y()
        self.coord_label.setText(f"X: {int(x)}, Y: {int(y)}")
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.black, 3)
        painter.setPen(pen)

        painter.drawLine(110,150,309,150)
        painter.drawLine(309,150,309,202)
        painter.drawLine(110,246,204,246)
        painter.drawLine(204,246,204,164)
        painter.drawLine(204,164,284,164)
        painter.drawLine(284,164,284,200)
        painter.drawLine(330,202,330,172)
        painter.drawLine(330,172,349,172)
        painter.drawLine(349,172,349,201)
        
        painter.drawLine(390,204,390,114)
        painter.drawLine(390,114,489,114)
        painter.drawLine(489,114,489,152)
        painter.drawLine(109,351,217,351)
        painter.drawLine(217,351,217,180)
        painter.drawLine(217,180,368,180)
        painter.drawLine(368,180,368,201)
        painter.drawLine(509,151,509,122)
        painter.drawLine(509,122,671,123)
        painter.drawLine(671,123,671,218)
        painter.drawLine(671,218,706,218)
        painter.drawLine(93,29,706,29)
        painter.drawLine(706,29,706,399)
        painter.drawLine(706,399,79,399)
        painter.drawLine(62,30,17,30)
        painter.drawLine(17,30,17,352)
        painter.drawLine(17,352,50,352)
        painter.drawLine(17,246,50,246)
        painter.drawLine(17,148,50,148)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NotXorGateGUI()
    win.show()
    sys.exit(app.exec())
