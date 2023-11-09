# Image Object Segmentation

**Process**

Given an image with 4 tomatoes in, we need to extract the bounding boxes of them.

Preprocessing, Threesholding and Watershed segmentation.

![](img/20221209142507.bmp)

Once finished the watershed segmentation, we take the bigger regions, apply blur, threeshold and Laplacian Border detection with isodata threeshold.

![](img/20221209142538.bmp)

Use kmeans to get the centers of the tomatoes.

![](img/20221209152656.bmp)

And finally measure the diagonal of the tomatoes. 

![](img/20221209154944.bmp)

**How do use it?**

```sh
python examen2.py jitomates.jpg
...
Tus objetos de interés son oscuros : 1 : si, 0 : no => 1
...
Ingresa el número de regiones que ocupan tus objetos : 8
...
¿Cuántos objetos hay en la imagen?4
```