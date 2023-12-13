import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5 import QtWidgets, QtCore, QtGui
from PIL import ImageGrab, Image
from tensorflow import keras

# Neural Network using PyTorch and Keras
class AttractivenessClassifier(nn.Module):
    def __init__(self):
        super(AttractivenessClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load pre-trained Keras model for attractiveness prediction
keras_model = keras.models.load_model('path/to/your/keras/model')

class SnippingTool(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(0, 0, 800, 600)
        self.setWindowTitle('Snipping Tool Pro')
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.setWindowOpacity(0.3)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        print('Capture the screen...')
        self.show()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), 3))
        qp.setBrush(QtGui.QColor(128, 128, 255, 128))
        qp.drawRect(QtCore.QRect(self.begin, self.end))

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.close()

        x1, y1 = min(self.begin.x(), self.end.x()), min(self.begin.y(), self.end.y())
        x2, y2 = max(self.begin.x(), self.end.x()), max(self.begin.y(), self.end.y())

        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        img.save('capture.png')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

  l
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

  
        pytorch_model = AttractivenessClassifier()
        pytorch_model.load_state_dict(torch.load('path/to/your/pytorch/model', map_location=torch.device('cpu')))
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_prediction = pytorch_model(img_tensor).item()

    
        keras_prediction = keras_model.predict(np.array([img_tensor.numpy()]))[0][0]

     
        self.display_results(pytorch_prediction, keras_prediction)

    def display_results(self, pytorch_prediction, keras_prediction):
        result_str = f"PyTorch Prediction: {pytorch_prediction:.4f}\nKeras Prediction: {keras_prediction:.4f}"
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Prediction Results")
        msg_box.setText(result_str)
        msg_box.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    snipping_tool = SnippingTool()
    snipping_tool.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())
