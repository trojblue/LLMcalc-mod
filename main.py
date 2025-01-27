import sys
from PySide6.QtWidgets import QApplication
from ui import LLMCalculatorUI

def main():
    app = QApplication(sys.argv)
    window = LLMCalculatorUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
