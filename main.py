import sys
from core.utils.config import Config
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app_config = Config()
    window = MainWindow(app_config)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()