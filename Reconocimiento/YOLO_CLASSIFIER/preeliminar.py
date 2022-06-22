import cv2
import numpy as np
import pytesseract
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import imutils
import os
import sys

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carga del algoritmo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = ["license_plate"]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


img = cv2.imread('imagenes/2.jpg')
height, width, channels = img.shape

# Deteccion de los objetos
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
imgH, imgW, _ = img.shape

net.setInput(blob)
outs = net.forward(output_layers)

# Muestra de resultados
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Objeto ha sido detectado con exito
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordenadas del rectangulo creado
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)

margen = 4
numero_matricula = ""

# Coordenadas en las que se encuentra el objeto en la imagen
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        placa = img[y-margen:y+h+margen,x-margen:x+w+margen,:]
        numero_matricula = pytesseract.image_to_string(placa,lang='eng')
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)


cv2.imshow('Imagen', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
