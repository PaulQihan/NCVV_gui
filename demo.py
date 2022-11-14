
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from test_decode import ncvv

class QTWindow(QWidget):
    def __init__(self):
        self.model = ncvv()
        self.model.render()
        print(self.model.rgb)
        super(QTWindow, self).__init__()
        
        self.mouse_move = False
        self.initialize_model()

        # Size of Window
        self.resize(400, 500)

        # Title
        self.label_title = QLabel("NCRF Demo", self)
        self.label_title.setFont(QFont('Arial Black', 20))
        self.label_title.setAlignment(Qt.AlignCenter)

        # Cover
        self.cover = QImage(self.model.rgb, self.model.rgb.shape[1], self.model.rgb.shape[0], self.model.rgb.shape[1]*3, QImage.Format_RGB888)
        # self.pixmap = QPixmap('cover.JPG')
        self.pixmap = QPixmap.fromImage(self.cover).scaled(300, 300)
        self.label_image = QLabel(self)
        self.label_image.setPixmap(self.pixmap)
        self.label_image.setAlignment(Qt.AlignCenter)

        # Time Slider
        self.label_t = QLabel("t :  0.0", self)
        self.label_t.setFont(QFont('Arial Black', 10))
        self.label_t.setAlignment(Qt.AlignLeft)
        self.slider_t = QSlider(Qt.Horizontal, self)
        self.slider_t.setRange(
            int(self.t_min / self.t_single_step), int(self.t_max / self.t_single_step))
        self.slider_t.setSingleStep(1)
        self.slider_t.valueChanged.connect(
            lambda: self.slider_change_func(self.slider_t))
        self.slider_t.setValue(self.time)
        # self.label_aperture_mode = QLabel("Aperture Mode :", self)
        # self.label_aperture_mode.setFont(QFont('Arial Black', 10))
        # self.label_aperture_mode.setAlignment(Qt.AlignLeft)

        self.layout_body = QVBoxLayout()
        self.layout_top = QHBoxLayout()
        self.layout_middle = QHBoxLayout()
        self.layout_bottom = QHBoxLayout()
        self.layout_output = QVBoxLayout()
        self.layout_output.setAlignment(Qt.AlignTop)
        self.layout_body.addLayout(self.layout_top)
        self.layout_top.addWidget(self.label_title)
        self.layout_body.addLayout(self.layout_middle)
        self.layout_body.addLayout(self.layout_bottom)

        self.layout_middle.addWidget(self.label_image)
        self.layout_middle.addLayout(self.layout_output)
        self.layout_bottom.addWidget(self.label_t)
        self.layout_bottom.addWidget(self.slider_t)

        self.label_fps = QLabel("fps :  " + str(round(self.fps, 1)), self)
        self.label_fps.setFont(QFont('Arial Black', 10))
        self.label_fps.setAlignment(Qt.AlignLeft)
        self.layout_output.addWidget(self.label_fps)

        self.label_move_x = QLabel(
            "move_x :  " + str(round(self.move_x, 1)), self)
        self.label_move_x.setFont(QFont('Arial Black', 10))
        self.label_move_x.setAlignment(Qt.AlignLeft)
        self.layout_output.addWidget(self.label_move_x)

        self.label_move_y = QLabel(
            "move_y :  " + str(round(self.move_y, 1)), self)
        self.label_move_y.setFont(QFont('Arial Black', 10))
        self.label_move_y.setAlignment(Qt.AlignLeft)
        self.layout_output.addWidget(self.label_move_y)

        self.setLayout(self.layout_body)
        print("initializing pyqt done")
        

    def initialize_model(self):
        self.time = 0
        self.t_min = 0
        self.t_max = 15
        self.t_single_step = 0.1
        self.fps = 0
        self.move_x = 0
        self.move_y = 0

        pass

    def update_model(self):
        #############################################################################
        # We will change the output image of the model here.
        #############################################################################
        self.model.update_pose(self.move_x, self.move_y, self.time)
        self.model.render()
        self.cover = QImage(self.model.rgb, self.model.rgb.shape[1], self.model.rgb.shape[0], self.model.rgb.shape[1]*3, QImage.Format_RGB888)
        # self.pixmap = QPixmap('cover.JPG')
        self.pixmap = QPixmap.fromImage(self.cover).scaled(300, 300)
        self.label_image.setPixmap(self.pixmap)
        self.label_image.setAlignment(Qt.AlignCenter)

        # Load output data
        if self.time == 0:
            self.label_t.setText("t :  0.0")
        else:
            self.label_t.setText("t : " + str(self.time))
        self.label_fps.setText("fps : " + str(self.fps))
        self.label_move_x.setText("move_x : " + str(self.move_x))
        self.label_move_y.setText("move_y : " + str(self.move_y))

    def slider_change_func(self, slider):
        if slider == self.slider_t:
            self.time = self.slider_t.value() / 10
            self.update_model()
        # self.system.get_image()
        # self.update_image()

    def mouseReleaseEvent(self, evt):
        self.move_Flag = False
        self.move_x = 0
        self.move_y = 0
        self.update_model()

    def mousePressEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            self.move_Flag = True
            self.mouse_x = evt.globalX()
            self.mouse_y = evt.globalY()

    def mouseMoveEvent(self, evt):
        if self.move_Flag:
            self.move_x = evt.globalX() - self.mouse_x
            self.move_y = evt.globalY() - self.mouse_y
            self.update_model()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = QTWindow()
    demo.show()

    sys.exit(app.exec_())
