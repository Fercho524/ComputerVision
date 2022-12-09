import cv2 
import numpy as np
from sys import argv 

# Imagen en Escala de Grises
image = cv2.imread(argv[1])
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
m,n = image.shape

# Reducción de ruido
image = cv2.GaussianBlur(image,(5,5),1)

# Segmentación por umbral, se obtienen los elementos claros u obscuros
from umbrales import isoData

T, img_bin = isoData(image,80)
invertir = int(input("Tus objetos de interés son oscuros : 1 : si, 0 : no => "))
img_bin = 255 - img_bin if invertir==1 else img_bin

# Segmenación Watershed
from watershed import Watershed,watershed_colors

regiones = Watershed(img_bin)
regiones_coloreadas = watershed_colors(regiones)

# Obtención de las regiones más grandes
from umbrales import Histogram

hist = Histogram(regiones)
hist = [[hist[k],k] for k in range(0,256)]
hist.sort(reverse=True)

cv2.imshow("Original",image)
cv2.imshow("Jitomates",img_bin)
cv2.imshow("Regiones",regiones)
cv2.imshow("RegionesColoreadas",regiones_coloreadas)

cv2.waitKey()
cv2.destroyAllWindows()

# Se seleccionan las n regiones más grandes
numRegions = int(input("Ingresa el número de regiones que ocupan tus objetos : ")) + 1 # 8 es el número correcto
topRegions = np.array(hist[0:numRegions])
topRegions = topRegions[:,1] 

# Se crea una imagen binaria con los objetos de interés
jitomates = np.zeros((m,n),dtype=np.uint8)

for i in range(0,m):
    for j in range(0,n):
        if regiones[i][j] in topRegions and regiones[i][j]!=0:
            jitomates[i][j]=255
        else:
            jitomates[i][j]=0

# Se elimina el ruido para quitar los cortes entre las regiones
from umbrales import Umbralizar

jitomates_cortados = jitomates.copy()
jitomates = cv2.GaussianBlur(jitomates,(51,51),5)
clean_tomatoes = Umbralizar(jitomates,127)

# Detección de bordes con filtro LoG
from laplace import borderDetection

borders = borderDetection(clean_tomatoes)

cv2.imshow("JitomatesCortados",jitomates_cortados)
cv2.imshow("Jitomates",jitomates)
cv2.imshow("JitomatesLimpios",clean_tomatoes)
cv2.imshow("Bordes",borders)

cv2.waitKey()
cv2.destroyAllWindows()


# Agrupamiento de coordenadas
from clustering import Kmeans 

# Añadimos las coordenadas de los bordes
white_coords = []

for i in range(m):
    for j in range(n):
        if borders[i,j]==255:
            white_coords.append([i,j])

# Agrupamos las coordenadas de los bordes, la cantidad de clusters es la cantidad de objetos
# , los centroides estarán en el centro del objeto y este valor se pregunta al usuario.
nobj = int(input("¿Cuántos objetos hay en la imagen?"))
coordenadas = np.array(white_coords)

model = Kmeans(nobj,20)
model.fit(coordenadas)

print(f"LEN : {len(white_coords)} vs {len(model.labels)} Labels = {np.unique(model.labels)}")

# Creamos una lista para guardas las coordenadas de cada cluster
cluster_coords = []

for c in range(nobj):
    cluster_coords.append([])

# Guardamos las coordenadas de cada cluster
for i in range(len(white_coords)):
    k = model.labels[i]
    cluster_coords[k].append(white_coords[i])

# Creamos tantas imágenes como objetos hayan sido detectados
objects = []
bounds = [] # Guarda las regiones de los objetos [x0,y0,x1,y1]

for obj_coords in cluster_coords:
    coords = np.array(obj_coords)
    xmin = coords[np.argmin(coords[:,0]),0]
    xmax = coords[np.argmax(coords[:,0]),0]
    ymin = coords[np.argmin(coords[:,1]),1]
    ymax = coords[np.argmax(coords[:,1]),1]

    bounds.append([xmin,ymin,xmax,ymax])

    print(f"Objeto encontrado desde el punto (x={xmin} , y={ymin} ) a ( x={xmax} , y={ymax} )")

    ni = ymax- ymin
    mi = xmax-xmin

    obj_img = np.zeros((mi,ni))
    print(f"Creando imagen con dimensiones {mi} x {ni}")

    for i in range(mi):
        for j in range(ni):
            obj_img[i,j] = borders[i+xmin,j+ymin]

    objects.append(obj_img)

# Graficando las líneas
marked_image = cv2.imread(argv[1])

for o in range(len(objects)):
    x,y,w,h = bounds[o]
    print(f"Trazando una línea de [ {x} , {y} ] a [ {w} , {h} ]")
    cv2.line(marked_image,(y,x),(h,w),color=(0,0,255),thickness=2)
    longitud_linea = np.sqrt((x-w)**2 +(y-h)**2)
    print(f"La longitud de la línea es : {longitud_linea}")
    xmed = int((y+h)/2)
    ymed = int((x+w)/2)
    cv2.putText(marked_image,f"L={int(longitud_linea)}",(xmed,ymed),cv2.FONT_HERSHEY_SIMPLEX,0.5,color=(255,255,255),thickness=2)


# Resultados
cv2.imshow("Bordes",borders)

for i in range(len(objects)):
    cv2.imshow(f"Object{i}",objects[i])

cv2.imshow("ConLineas",marked_image)

cv2.waitKey()
cv2.destroyAllWindows()
