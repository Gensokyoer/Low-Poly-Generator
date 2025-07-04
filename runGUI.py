import sys
import main
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog,
                               QMessageBox, QGraphicsScene, QGraphicsView)
from gui.app6 import LowPolyApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = LowPolyApp()
    window.setWindowTitle("低多边形图像生成器")
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())