# Examen 2 Castro Mendieta Fernando 5BV1 

Cómo usar?

```sh
python examen2.py jitomates.jpg
...
Tus objetos de interés son oscuros : 1 : si, 0 : no => 1
...
Ingresa el número de regiones que ocupan tus objetos : 8
...
¿Cuántos objetos hay en la imagen?4
```

Para esta imagen en particular, se debe ingresar 8 regiones de segmentación y 4 objetos a detectar

Preprocesamiento de la imagen y segmentación básica

![](img/20221209142507.bmp)

Detección de bordes de los objetos más grandes 

![](img/20221209142538.bmp)

Segmentación de objetos usando K-Means

![](img/20221209152656.bmp)

Obtención de las distancias y medida de las diagonales de cada objeto. 

![](img/20221209154944.bmp)