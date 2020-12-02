#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 11 18:04:55 2020

@author: felipe
"""

import multiprocessing
import sys
import cv2
import numpy as np
import os
import shutil
import time


# Parte inicial, obtener los directorios que contiene las imagenes de sub apertura de los campos de luz
print("\nEl directorio de trabajo es: " + os.getcwd())

HR_dirs = "Campos_de_luz/ajedrez"
print ("\nLos directorios que contienen cada campo de luz HR son : " + str(HR_dirs)) 

# Se recorre cada director contenedor de un campo de luz
# NOTA: 'frame' hace referencia a fotograma o a cada imagen de sub-apertura.
for i in range(len(HR_dirs)):

    # Marcador de tiempo inicial para supervisar la duracion de cada asignacion
    start_time_aux = time.time()

    # Se obtiene una lista con los nombres de archivos de las imagenes de sub-apertura
    filelist=os.listdir(HR_dirs)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")): # Remueve nombres de archivos que no sean .png
            filelist.remove(fichier)
    filelist.sort()
    
    print("\nRecorriendo el directorio : " + str(HR_dirs))

    # Se obtienen las dimensiones de las imagenes de sub-apertura
    img = cv2.imread((HR_dirs + '/'+filelist[0]))
    h,w = img.shape[0:2]

    # El campo de luz tiene 8x8 vistas
    big_img = np.zeros((h*8, w*8, 3), dtype=np.uint8)

    jx = 0
    jy = 0
    for filename in filelist:

        print("Voy en la imagen: %s" % (str(filename)))

        img = cv2.imread((HR_dirs + '/' + filename))
        h,w = img.shape[0:2]

        posy = 0 + jy
        posx = 0 + jx

        for row in range(h):
            for col in range(w):
                
                big_img[posy,posx,:] = img[row,col,:]

                posx = col*8 + jx
            posy = row*8 + jy

        if jx == 7:
            jy += 1
            jx = 0
        else:
            jx += 1        
                
    cv2.imwrite(('Campos_de_luz/ajedrez/campo_de_luz_ajedrez.png'), big_img)
    
    print("---Tiempo de ejecucion: %s segundos ---" % (time.time() - start_time_aux))
    print("\nSe ha almacenado correctamente la imagen grande")
