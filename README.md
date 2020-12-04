# Procesamiento-de-Campos-de-Luz

Este repositorio contiene el código fuente para los algoritmos enunciados a continuación, desarrollados en la asignatura de enfásis "Procesamiento de Imágenes y Video" en el programa de Ingeniería Electrónica de la Pontificia Universidad Javeriana. ***Autores: Juan Felipe Jaramillo Hernández, María Fernanda Hernández Baena, Valeria Méndez Sarmiento.***

Se implementaron los siguientes algoritmos para el post-procesamiento de un campo de luz capturado; Se implementa un algoritmo para extraer las [imágenes de sub-apertura](Extraccion_imgs_sub_apertura/obtencion_imgs_subapertura_final.py) de un campo de luz capturado con una cámara plenóptica afocal, una vez se obtienen las imágenes de sub-apertura, se implementa un algoritmo computacional para [sub-muestrear](Sub_muestreo/decimacion_espacial.py) espacial y angularmente el campo de luz, además, se implementa otro algoritmo para organizar el campo de luz en una [secuencia sin discontinuidades](Secuencia_sin_discontinuidades/ssd_puntos_ping_pong_Multiprocessing.py) utilizando el flujo óptico. El video y el artículo se pueden encontrar acá:

Video: <>
Artículo: <https://bit.ly/36EnkNd>

Tabla de Contenidos
=================

<!--ts-->
* [Procesamiento de Campos de Luz](#Procesamiento-de-Campos-de-Luz)
* [Tabla de Contenidos](#tabla-de-contenidos)
* [Resultados Obtenidos](#Resultados-Obtenidos)
* [Referencias](#Referencias)
<!--te-->

### Resultados Obtenidos
A partir del siguiente campo de luz (capturado por una cámara plenóptica afocal de alta resolución) se logró extraer la vista central de todos los micro-lentes. El sensor de imagen consta de 220 x 176 micro-lentes, y una pequeña porción del campo de luz se puede observar a continuación:

**Campo de luz capturado (*Imagen propia*)**

![Campo de luz HR](Extraccion_imgs_sub_apertura/LF_dartthvader_proc.png)

**Vista central del campo de luz capturado, 220 x 176 (*Imagen propia*)**

![vista central - Campo de luz HR](Extraccion_imgs_sub_apertura/darthvader_Processed_img11_VistaCentral_Completa1x1.png)

Ya que la imagen anterior tiene una resolución pequeña (de acuerdo a la matriz de micro-lentes de la cámara plenóptica), se realiza una interpolación bicúbica sobre el resultado anterior:

**Vista central interpolada del campo de luz capturado (*Imagen propia*)**

![vista central - Campo de luz HR - interpolada](Extraccion_imgs_sub_apertura/darthvader_Processed_img11_VistaCentral_Completa_interpolada.png)

Los resultados obtenidos anteriormente fueron los esperados, debido a la estructura de la malla de micro-lentes y la tecnología de la cámara plenóptica con la que se capturó. Sin embargo, dada la resolución obtenida, los siguientes resultados se obtuvieron de un campo de luz tomado del [Conjunto de datos de Campos de Luz de Stanford](http://lightfield.stanford.edu/lfs.html).

A continuación, se muestra la secuencia sin discontinuidades obtenida después de implementar una organización de acuerdo al flujo óptico en las dimensiones X y Y entre las imágenes de sub-apertura del campo de luz, implementado en el [algoritmo](Secuencia_sin_discontinuidades/ssd_puntos_ping_pong_Multiprocessing.py).

**Campo de luz organizado en secuencia sin discontinuidades (*Campo de luz tomado de [1]*)**

![secuencia sin discontinuidades - Campo de luz HR](Secuencia_sin_discontinuidades/video_sin_discontinuidades.gif)

Las imágenes de sub-apertura del campo de luz anterior son sub-muestreadas espacialmente por un factor K = 4, implementando el siguiente [algoritmo](Sub_muestreo/decimacion_espacial.py).

**Imagen de sub-apertura del campo de luz (*Campo de luz tomado de [1]*)**

![Imagen sub-apertura Campo de luz HR](Sub_muestreo/Campos_de_luz/ajedrez/out_04_05_-797.908081_-3330.062744_.png)

**Imagen de sub-apertura del campo de luz sub-muestreada espacialmente (*Campo de luz tomado de [1]*)**

![Imagen sub-apertura Campo de luz HR](Sub_muestreo/ajedrez_submuestreados/LR_out_04_05_-797.908081_-3330.062744_.png)

Para el sub-muestreo angular, se utilizó el siguiente [algoritmo](Sub_muestreo/creacion_png_campo_de_luz.py) para generar una imagen espacial del campo de luz completo.

**Porción del campo de luz espacialmente completo (*Campo de luz tomado de [1]*)**

![Campo de luz HR completo](Sub_muestreo/Campos_de_luz/ajedrez/campo_de_luz_8x8.png)

La imagen completa anterior se sub-muestreó angularmente con el mismo [algoritmo](Sub_muestreo/decimacion_espacial.py) implementado anteriormente para sub-muestrear espacialmente las imágenes de sub-apertura. Una porción del resultado se muestra anterior:

**Porción del campo de luz sub-muestreado angularmente (*Campo de luz tomado de [1]*)**

![Campo de luz HR sub-muestreado](Sub_muestreo/ajedrez_submuestreados/campo_de_luz_2x2.png)

### Referencias

[1] Conjunto de datos de Campos de Luz, Universidad de Stanford. Disponible en http://lightfield.stanford.edu/lfs.html