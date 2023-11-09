import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from sys import argv
from watershed import *
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D

from utils import VERBOSE






# ###################################################
# Segmentación R, G, B
# ###################################################

def channel_segmentation(image):
    VERBOSE and print("Separando por canales de color")
    m, n = [image.shape[0], image.shape[1]]
    zeros = np.zeros((m, n), dtype=np.uint8)

    blue = cv2.merge([
        image[:, :, 0], 
        zeros, 
        zeros ]
    )

    green = cv2.merge([
        zeros, 
        image[:, :, 1],
        zeros, ]
    )

    red = cv2.merge([
        zeros, 
        zeros, 
        image[:, :, 2] ]
    )

    return [blue,green,red]

def thresold_segmentation(image):
    VERBOSE and print("Segmentando por umbral")

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T, biImage = isoData(imageGray, 120)

    VERBOSE and print(f"El umbral es : {T}")
    
    graySegmented = cv2.merge([
        cv2.bitwise_and(biImage, image[:, :, 0]),
        cv2.bitwise_and(biImage, image[:, :, 1]),
        cv2.bitwise_and(biImage, image[:, :, 2]),
    ])

    return graySegmented



def euclidean(v,w):
    return np.sqrt(sum([(v[i]-w[i])**2 for i in range(0,len(w))]))

class Kmeans:
    def __init__(self, nclusters, iterations=100):
        self.n_clusters = nclusters
        self.max_iter = iterations
        self.labels = []

    def init_centroids(self, points):
        return random.sample(points.tolist(),self.n_clusters)

    def calc_centroids(self, points, labels):
        centroids = np.zeros((self.n_clusters, points.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(points[labels == k, :], axis=0)
        return centroids

    def matrix_distance(self, points, centroids):
        distance = np.zeros((points.shape[0], self.n_clusters))
        
        for i in range(len(points)):
            for k in range(self.n_clusters):
                distance[i][k] = euclidean(centroids[k],points[i])

        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def fit(self, points):
        self.centroids = self.init_centroids(points)

        for i in range(self.max_iter):
            VERBOSE and print(f"Trainning model stage {i}")
            old_centroids = self.centroids
            distance = self.matrix_distance(points, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.calc_centroids(points, self.labels)
            
            if np.all(old_centroids == self.centroids):
                break

    def predict(self, point):
        distance = self.matrix_distance(point, self.centroids)
        return self.find_closest_cluster(distance)

def plotclusters(nclusters,data,model):
    fig = plt.figure()
    ax = Axes3D(fig)

    for k in range(nclusters):
        labeldata = data[model.labels == k]
        ax.scatter(labeldata[:,0],labeldata[:,1],labeldata[:,2])

    ax.scatter(model.centroids[:,0],model.centroids[:,1],model.centroids[:,2],marker="*")
    plt.show()


def kmeans_segmentation(image,nclusters,niter):
    m,n = [image.shape[0],image.shape[1]]

    VERBOSE and print("Vectorizando la imagen")
    vectorized = np.float32(image.reshape((m*n,3)))

    VERBOSE and print("Usando K-means para agrupar")
    model = Kmeans(nclusters,niter)
    model.fit(vectorized)

    VERBOSE and print("Centroides encontrados")
    VERBOSE and print(model.centroids)

    resultimage = np.zeros((m,n,3),dtype=np.uint8)

    VERBOSE and print("Obteniendo imagen original")
    
    for i in range(m):
        for j in range(n):
            pix = np.array([image[i][j]])
            clster = model.predict(pix)[0]
            resultimage[i,j,:] = np.uint8(model.centroids[clster])

    return model.centroids, resultimage

if __name__ == "__main__":
    VERBOSE and print("Reading image")
    image = cv2.imread(argv[1])
    image = cv2.GaussianBlur(image, (5, 5), 1)

    cv2.imshow("Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # ###################################################
    # Lectura de imagen a color
    # ###################################################


    b,g,r = channel_segmentation(image)

    cv2.imshow("Blue", b)
    cv2.imshow("Green", g)
    cv2.imshow("Red", r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ###################################################
    # Segmentación por Umbral
    # ###################################################

    graySegmented = thresold_segmentation(image)

    cv2.imshow("Original", image)
    cv2.imshow("GraySegmented", graySegmented)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # ###################################################
    # KMEANS - FUNCTIONS
    # ###################################################
    nclusters = int(input("Ingresa el número de clusters :"))
    centroids,ksegmented = kmeans_segmentation(image,nclusters,10)

    # cv2.imshow("Original", image)
    # cv2.imshow("Segmented", ksegmented)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


