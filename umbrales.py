import numpy as np
import cv2

from utils import VERBOSE

def Histogram(image):
    m, n = [image.shape[0], image.shape[1]]
    hist = np.zeros((256))

    for i in range(0, m):
        for j in range(0, n):
            p = image[i][j]
            hist[p] += 1

    return hist


def Umbralizar(image, T):
    m, n = [image.shape[0], image.shape[1]]
    result = np.zeros((m, n))

    for i in range(0, m):
        for j in range(0, n):
            if (image[i][j] > T):
                result[i][j] = 255
            else:
                result[i][j] = 0

    return np.uint8(result)


def OTSU(image):
    histogram = Histogram(image)
    variances = []

    m, n = [image.shape[0], image.shape[1]]
    print(histogram)

    for T in range(0, 256):
        Wb = sum(histogram[0:T]) / (m*n)
        Wf = sum(histogram[T:255]) / (m*n)

        muB = sum([histogram[i]*i for i in range(0, T)]) / (sum(histogram[0:T])+1)
        muF = sum([histogram[i]*i for i in range(T, 256)]) / (sum(histogram[T:255])+1)

        varB = sum([histogram[k]*(k-muB)**2 for k in range(0, T)]) / (sum(histogram[0:T])+1)
        varF = sum([histogram[k]*(k-muF)**2 for k in range(T, 256)]) / (sum(histogram[T:255])+1)

        otsu = Wb*varB + Wf*varF
        variances.append(otsu)

        if (VERBOSE):
            print(f"Wb = {Wb}, muB = {muB}, varB = {varB}")
            print(f"Wf = {Wf}, muF = {muF}, varF = {varF}")
            print(f"Ow = {otsu}")

    variances = np.array(variances)
    u = variances.argmin()

    if VERBOSE:
        print(f"Umbral obtenido = {u}")
        print(variances)
    return T,Umbralizar(image, int(u))


def picos(image):
    histogram = np.array(Histogram(image))
    variances = np.zeros((256))

    m1Index = histogram.argmax()
    m1 = histogram[m1Index]

    if VERBOSE:
        print(f"El primer pico está en {m1Index}")

    for k in range(0,256):
        variances[k]=histogram[k] * (k-m1)**2 
    
    m2Index = variances.argmax()
    T = int((m1Index+m2Index)/2)

    if VERBOSE:
        print(f"El segundo pico está en : {m2Index}")

    return T,Umbralizar(image,T)


def isoData(image,inicial):
    T1,T2 = [inicial, inicial]

    delta = 255
    histogram = Histogram(image)

    G1,G2 = [0,0]
    m1,m2 = [0,0]

    while (delta>1):
        T1 = T2

        if T1<0:
            T1=0

        G1 = sum(histogram[0:T1+1])
        G2 = sum(histogram[T1:255])

        m1 = sum([k*histogram[k] for k in range(0,T1)]) / (G1+1)
        m2 = sum([k*histogram[k] for k in range(T1,255)]) / (G2+1)

        T2 = int(abs((m1+m2)/2))
        delta = T2-T1

        if VERBOSE:
            print(f"G1 = {G1} , G2 = {G2} , m1 = {m1}, m2 = {m2} ==> T2 = {T2}")

    T = int(T2)
    
    if VERBOSE:
        print(f"Umbral encontrado con ISODATA: {T}")
    return T,Umbralizar(image,T)


if __name__ == "__main__":
    image = cv2.imread("img/Cameraman.png")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(5,5),1)

    T1,U1 = isoData(image,80)
    T2,U2 = OTSU(image)
    T3,U3 = picos(image)

    cv2.imshow("Original",image)
    cv2.imshow("Isodata",U1)
    cv2.imshow("Otsu",U2)
    cv2.imshow("Picos",U3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()