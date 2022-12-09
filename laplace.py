import os
import cv2
import math
import numpy as np
from termcolor import cprint
from sys import argv


def LoG(x, y, sigma):
    t1 = 1/(math.pi*sigma**4)
    t2 = 1 - ((x**2 + y**2) / (2*sigma**2))
    t3 = math.exp((-(x**2 + y**2))/(2*sigma**2))
    return t1*t2*t3


def LoGKernel(n, sigma):
    print(f"Calculando kernel Hlog(x,y) de tamaño {n} y sigma = {sigma}")
    kernel = np.zeros((n, n))

    for i in range(int(-((n - 1) / 2)), int(((n - 1) / 2)+1)):
        for j in range(int(-((n - 1) / 2)), int(((n - 1) / 2)+1)):
            kernel[i][j] = LoG(i, j, sigma)
            j += 1
        i += 1

    print(kernel)
    return kernel


def ZeroCrossing(img):
    result = np.zeros(img.shape)
    m, n = [img.shape[0], img.shape[1]]

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            negatives = 0
            positives = 0

            # Obteniendo los vecinos en una vecindad 3x3
            neighbour = [img[i+1, j-1], img[i+1, j], img[i+1, j+1], img[i,
                                                                        j-1], img[i, j+1], img[i-1, j-1], img[i-1, j], img[i-1, j+1]]

            # Maximo y mínimo de la vecindad
            maxN = max(neighbour)
            minN = min(neighbour)

            # Se obtienen los elementos distintos de cero
            for neigh in neighbour:
                if neigh > 0:
                    positives += 1
                elif neigh < 0:
                    negatives += 1

            # Detectando subidas o bajadas
            if negatives > 0 and positives > 0:
                if img[i, j] > 0:
                    result[i, j] = img[i, j] + np.abs(minN)
                elif img[i, j] < 0:
                    result[i, j] = np.abs(img[i, j]) + maxN

    # Normalizando la imagen
    normalized = (result/result.max()) * 255
    result = np.uint8(normalized)

    return result


simpleKernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

from umbrales import isoData


def borderDetection(image):
    result = cv2.GaussianBlur(image,(5,5),1.5)
    result = cv2.filter2D(image,-1,simpleKernel)
    T,result = isoData(result,10)
    return result


# # Image Reading
# src = argv[1]
# nombre, extension = os.path.splitext(src)
# dest = nombre.split("/")[-1]+"borders.png"

# cprint(f"Leyendo la imagen {src}", "yellow")
# image = cv2.imread(src)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Preprocesing
# cprint("Aplicando eliminación de ruido gaussiana", "yellow")
# image = cv2.GaussianBlur(image,(5,5),1)

# # Reading Data
# # cprint("Aplicando kernel HloG", "yellow")
# # kernsize = int(input("Ingresa el tamaño del kernel : "))
# # sigma = float(input("Ingresa el sigma : "))

# # # LoG Filter
# # logkernel = LoGKernel(kernsize, sigma)
# # imglog = cv2.filter2D(image, cv2.CV_64F, logkernel)
# # imglog = imglog/imglog.max()

# # # Zero Crossing
# # cprint("Realizando la cruza por ceros", "yellow")
# # zeroCrossed = ZeroCrossing(imglog)

# # # Final Edges
# # cprint("Getting the final borders", "yellow")
# # final = cv2.filter2D(zeroCrossed, -1, simpleKernel)

# from watershed import *

# kernelFiltered = cv2.filter2D(image,-1,simpleKernel)
# T,finalBorders =isoData(kernelFiltered,10)
# finalBorders = 255 - finalBorders
# # Results
# cv2.imshow("Original", image)
# # cv2.imshow("LOG", imglog)
# # cv2.imshow("LOGUMBRAL", zeroCrossed)
# # cv2.imshow("Final", final)
# cv2.imshow("ConKernel",kernelFiltered)
# cv2.imshow("Bordes",finalBorders)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #cprint("Image Saved", "green")
# #os.chdir("results")

# #cv2.imwrite(dest, zeroCrossed)
