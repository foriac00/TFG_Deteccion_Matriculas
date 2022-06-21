import cv2
import numpy as np
import pytesseract
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import imutils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = ["license_plate"]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
numero_matricula = ""
indexes = ""

def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])

    if len(path_image) > 0:
        global image

        # Leer la imagen de entrada y la redimensionamos
        image = cv2.imread(path_image)

        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=180)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)

        lblInputImage.configure(image=img)
        lblInputImage.image = img

        # Label IMAGEN DE ENTRADA
        lblInfo1 = Label(root, text="IMAGEN DE ENTRADA:")
        lblInfo1.grid(column=0, row=1, padx=5, pady=5)

        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        lblOutputImage.image = ""
        selected.set(0)

def ejecucion():
    global image
    global indexes
    global numero_matricula

    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    imgH, imgW, _ = image.shape

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
            placa = image[y - margen:y + h + margen, x - margen:x + w + margen, :]
            numero_matricula = pytesseract.image_to_string(placa, lang='eng')
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    imageToShowOutput = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Para visualizar la imagen en lblOutputImage en la GUI
    im = Image.fromarray(imageToShowOutput)
    img = ImageTk.PhotoImage(image=im)
    lblOutputImage.configure(image=img)
    lblOutputImage.image = img

    # Label IMAGEN DE SALIDA
    lblInfo3 = Label(root, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

def clasificar_por_años():
    global indexes
    global numero_matricula
    print('En la imagen hay', len(indexes), 'matricula/s')
    print("Matrícula :", numero_matricula)
    if numero_matricula[5] == 'B':
        print('Año 2003-2004')
    if numero_matricula[5] == 'C':
        print('Año 2003-2004')
    if numero_matricula[5] == 'D':
        print('Año 2004-2006')
    if numero_matricula[5] == 'F':
        print('Año 2006-2008')
    if numero_matricula[5] == 'G':
        print('2008-2010')
    if numero_matricula[5] == 'H':
        print('Año 2011-2014')
    if numero_matricula[5] == 'J':
        print('Año 2014-2017')
    if numero_matricula[5] == 'K':
        print('Año 2017-2019')
    if numero_matricula[5] == 'L':
        print('Año 2019-Actualidad')
    if numero_matricula[5] == ('A,E,I,O,U,Ñ'):
        print('Matricula no española')

####NO TOCAR
# Creamos la ventana principal
root = Tk()
# Label donde se presentará la imagen de entrada
lblInputImage = Label(root)
lblInputImage.grid(column=0, row=2)

# Label donde se presentará la imagen de salida
lblOutputImage = Label(root)
lblOutputImage.grid(column=1, row=1, rowspan=6)
# Label ¿Qué quieres hacer?
lblInfo2 = Label(root, text="¿Qué te gustaria hacer?", width=25)
lblInfo2.grid(column=0, row=3, padx=5, pady=5)
# Creamos los radio buttons y la ubicación que estos ocuparán
selected = IntVar()
rad1 = Radiobutton(root, text='Clasificar imagen', width=25,value=1, variable=selected, command= ejecucion)
rad2 = Radiobutton(root, text='Mostrar años',width=25, value=2, variable=selected, command= clasificar_por_años)
rad3 = Radiobutton(root, text='Salir',width=25, value=3, variable=selected, command= quit)

rad1.grid(column=0, row=4)
rad2.grid(column=0, row=5)
rad3.grid(column=0, row=6)
# Creamos el botón para elegir la imagen de entrada
btn = Button(root, text="Elige la imagen", width=25, command=elegir_imagen)
btn.grid(column=0, row=0, padx=5, pady=5)
#Texto de salida
text = Text(root, width=30, height=2)

root.mainloop()