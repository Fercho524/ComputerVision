from sys import argv
import numpy as np
import cv2

# Watershed
from skimage import morphology
from skimage.segmentation import watershed
from skimage.segmentation import random_walker
from skimage.feature import peak_local_max

from scipy import ndimage as ndi

from umbrales import * 

VERBOSE = True




def watershed_colors(labels):
    m,n = labels.shape

    # Se obtienen las distintas etiquetas
    labeledObjects = np.zeros((m,n,3),dtype=np.uint8)
    labelIDs = np.unique(labels)
    labelIDs = list(labelIDs)

    # Se generan colores aleatorios
    random_colors = []

    for c in labelIDs:
        b = np.random.randint(0,255)
        g = np.random.randint(0,255)
        r = np.random.randint(0,255)
        random_colors.append([b,g,r])

    # Dependiendo de la etiqueta se da un color a la nueva imagen
    for i in range(m):
        for j in range(n):
            cindex = labelIDs.index(labels[i][j])
            labeledObjects[i,j,:] = random_colors[cindex]
            if labels[i][j] == 0:
                labeledObjects[i,j,:] = [0,0,0]

    return labeledObjects

if __name__ == "__main__":
    # Lectura de imagen
    image = cv2.imread(argv[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Eliminación de ruido

    cv2.imshow("Original",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    nblur = int(input("Ingresa el tamaño del desenfoque :"))
    sigma = float(input("Ingresa el sigma :"))
    image = cv2.GaussianBlur(image,(nblur,nblur),sigma)

    # Binarización
    T1,binotsu = OTSU(image)
    T2,binpics = picos(image)
    T3,binisod = isoData(image,80)

    # Resultados de binarización
    if VERBOSE:
        cv2.imshow("Original",image)
        cv2.imshow("OTSU", binotsu)
        cv2.imshow("PICOS",binpics)
        cv2.imshow("ISODATA",binisod)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Selección de la mejor imagen binarizada
    print("Ingresa el tipo de umbral que quieras elegir:\n1: Otsu\n2: Picos\n3: Isodata:")
    preferenciaUmbral = int(input(""))
    preferenciaUmbral = 3

    if preferenciaUmbral==1:
        binarizada = binotsu
        umbral = T1
    elif preferenciaUmbral==2:
        binarizada = binpics 
        umbral = T2
    else:
        binarizada = binisod
        umbral = T3

    # Obtención del watershed labels
    distance = ndi.distance_transform_edt(binarizada)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binarizada)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binarizada)

    # Conversión de datos
    distance = np.array(distance,np.uint8)
    markers = np.array(markers,np.uint8)
    distancias = binarizada - distance
    labels = np.array(labels,dtype=np.uint8)

    # Objetos segmentados
    finalResult = watershed_colors(labels)

    # Resultados de segmentación
    cv2.imshow("Original",binarizada)
    cv2.imshow("Distancias",distance)
    cv2.imshow("ObjetosDetectados",distancias)
    cv2.imshow("Watershed",labels)
    cv2.imshow("ColoresEtiquetados",finalResult)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Watershed(bin_image):
    distance = ndi.distance_transform_edt(bin_image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=bin_image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=bin_image)

    distance = np.array(distance,np.uint8)
    markers = np.array(markers,np.uint8)
    distancias = bin_image - distance
    labels = np.array(labels,dtype=np.uint8)

    return labels
