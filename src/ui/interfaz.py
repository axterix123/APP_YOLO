import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
from ultralytics import YOLO

class LineDrawingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dibujo de líneas de conteo")
        self.setGeometry(100, 100, 800, 650)
        self.initUI()
        self.lines = []
        self.drawing = False
        self.current_line = []
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.line_color_index = 0

    def initUI(self):
        # Layout principal
        layout = QVBoxLayout()
        
        # Botones
        self.draw_button = QPushButton("Dibujar línea", self)
        self.draw_button.clicked.connect(self.toggle_draw_line)
        self.delete_button = QPushButton("Borrar línea", self)
        self.delete_button.clicked.connect(self.delete_line)
        self.save_button = QPushButton("Guardar líneas", self)
        self.save_button.clicked.connect(self.save_lines)
        self.start_button = QPushButton("Iniciar detección", self)
        self.start_button.clicked.connect(self.run_tracking)
        
        # Imagen
        self.image_label = QLabel(self)
        
        # Añadir widgets al layout
        layout.addWidget(self.image_label)
        layout.addWidget(self.draw_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.start_button)
        
        # Contenedor principal
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Cargar el primer frame del video
        self.load_first_frame()

    def load_first_frame(self):
        video_path = r"C:\Users\edwin\Documents\3 Personal\YOLO Project\YOLO\test\SS55-11-02-20231128-074311-20231128-075517.mp4"

        # Cargar el video y obtener el primer frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            height, width = frame.shape[:2]
            new_height = int((800 / width)*height)
            resized_frame = cv2.resize(frame, (800, new_height)) #NOTE: Change size in the future detecting size of video
            self.original_frame = resized_frame
            self.display_image(resized_frame)
        
    def run_tracking(self):
        model = YOLO(f"yolov10n.pt")
        video_path = r"C:\Users\edwin\Documents\3 Personal\YOLO Project\YOLO\test\SS55-11-02-20231128-074311-20231128-075517.mp4"

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                detection = model.track(frame, persist = True)
                annotated_frame = detection[0].plot()
                height, width = annotated_frame.shape[:2]
                new_height = int((800 / width)*height)
                resized_frame = cv2.resize(annotated_frame, (800, new_height)) #NOTE: Change size in the future detecting size of video
                self.original_frame = resized_frame
                self.display_image(resized_frame)
                cv2.waitKey(1)
            else:
                break
    
    def display_image(self, img):
        # Convertir la imagen de OpenCV a QImage
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_pixmap = QPixmap.fromImage(qt_image)
        
        # Mostrar la imagen en el QLabel
        self.image_label.setPixmap(qt_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def toggle_draw_line(self):
        if not self.drawing:
            self.drawing = True
            self.image_label.mousePressEvent = self.get_mouse_pos
            self.image_label.paintEvent = self.paintEvent
            self.current_line = []
        else:
            if len(self.current_line) == 2:
                self.lines.append(tuple(self.current_line))  # Almacenar como tupla
                self.current_line = []
                self.paintEvent()  # Volver a pintar para mostrar la línea dibujada
            
            self.drawing = False  # Cambiar el estado de dibujo después de completar una línea

    def get_mouse_pos(self, event):
        if self.drawing:
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                if 0 <= pos.x() <= self.image_label.width() and 0 <= pos.y() <= self.image_label.height():
                    self.current_line.append((pos.x(), pos.y()))  # Almacenar como tupla
                    if len(self.current_line) == 2:
                        self.toggle_draw_line() 

    def paintEvent(self, event=None):
        pixmap = QPixmap(self.image_label.pixmap())
        painter = QPainter(pixmap)

        for line in self.lines:
            start_point = QPoint(line[0][0], line[0][1])
            end_point = QPoint(line[1][0], line[1][1])
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawLine(start_point, end_point)

        if len(self.current_line) == 2:
            start_point = QPoint(self.current_line[0][0], self.current_line[0][1])
            end_point = QPoint(self.current_line[1][0], self.current_line[1][1])
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawLine(start_point, end_point)

        painter.end()
        self.image_label.setPixmap(pixmap)

    def delete_line(self):
        self.drawing = True
        # Implementar lógica para borrar una línea en la imagen
        # ...

    def save_lines(self):
        print("Coordenadas de las líneas guardadas:")
        for i, line in enumerate(self.lines):
            print(f"Línea {i+1}: Punto inicial {line[0]}, Punto final {line[1]}")
        print("Guardado exitoso.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LineDrawingWindow()
    window.show()
    sys.exit(app.exec_())