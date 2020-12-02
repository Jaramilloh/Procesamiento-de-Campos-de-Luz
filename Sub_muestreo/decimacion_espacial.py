#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:04:55 2020

@author: felipe
"""


import sys
import cv2
import numpy as np
import os
import shutil
import time


def borde_impar(img):
    '''
    Funcion para añadir bordes en los extremos de la imagen para que esta tenga dimensiones
    impares
    '''
    h,w = img.shape[0:2]

    while h%2 == True:
        img_aux = img.copy()
        b = np.zeros((1, len(img[0,:], 3)))
        img = np.concatenate((img_aux, b), axis=0)
        h, w = img.shape[0:2]

    while w%2 == True:
        img_aux = img.copy()
        b = np.zeros((len(img[:,0], 1, 3)))
        img = np.concatenate((img_aux, b), axis=1)
        h, w = img.shape[0:2]

    return img


def filtro_gaussian(img, sigmax, sigmay):
    '''
    Esta funcion genera el filtro Gaussiano en las dimensiones de la
    imagen a filtrar, con la frecuencia de corte especifica.

    Entradas:
        img: imagen a filtrar
        K: factor de sub-muestreo
        sigmax: desviacion estandar (frecuencia de corte) en la dimension 'x'
        sigmay: desviacion estandar (frecuencia de corte) en la dimension 'y'
    Salidas:
        gaussian_norm: filtro Gaussiano en frecuencia
    '''
    kx=img.shape[0] # Tamaño de la ventana en x
    ky=img.shape[1] # Tamaño de la ventana en y

    # Generación de la ventana del filtro Gaussiano en espacio de 'kx'x'ky'
    # con la frecuencia de corte espacial dada por 'sigmax' y 'sigmay'
    x=np.float32(cv2.getGaussianKernel(kx,sigmax))
    y=np.float32(cv2.getGaussianKernel(ky,sigmay))
    gaussian=x*y.T

    # Normalizar min-max el filtro en espacio para pasarlo a frecuencia
    gauss_min = np.min(gaussian)
    gauss_max = np.max(gaussian)
    gaussian_norm = (gaussian - gauss_min)/(gauss_max - gauss_min)

    # Se crean dos matrices con el mismo filtro, uno para multiplicar la parte real, y otro para multiplicar la parte imaginaria
    gausss = np.zeros((gaussian_norm.shape[0], gaussian_norm.shape[1], 2), dtype=np.float32)
    #gausss[:,:,0] = 1
    gausss[:,:,0] = gaussian_norm
    gausss[:,:,1] = gaussian_norm
    #gausss[:,:,1] = 1

    return gausss

def iteraciones_opt(img, gaussian, sfft_img_c1, sfft_img_c2, sfft_img_c3, K):
    '''
    Esta funcion evalua los 'n' filtros Gaussianos especificados en el
    numero de iteraciones. Se filtrar la imagen especifica, se sub-muestrea
    esoacialmente y se escala o sobre-muestrea de nuevo a partir de 
    una interpolacion bicubica para calcular un error medio absoluto 
    entre esta y la imagen original sin filtrar y sin sub-muestrear.

    Entradas:
        img: imagen a filtrar
        gaussian: filtro Gaussiano en frecuencia a probar
        sfft_img_c1: canal B  en frecuencia de la imagen a filtrar
        sfft_img_c2: canal G  en frecuencia de la imagen a filtrar
        sfft_img_c3: canal R  en frecuencia de la imagen a filtrar
        K: factor de sub-muestreo
    Salidas:
        J: costo entre la imagen filtrada sub-muestreara y escalada y 
        la imagen orginal
        img_dwn: imagen filtrada sub-muestreara
    '''
    # Multiplicación del filtro y cada canal de color en frecuencia
    img_filteredc1 = sfft_img_c1*gaussian
    img_filteredc2 = sfft_img_c2*gaussian
    img_filteredc3 = sfft_img_c3*gaussian

    # Inverse DFT shift
    img_fltc1_aux = np.fft.ifftshift(img_filteredc1)
    img_fltc2_aux = np.fft.ifftshift(img_filteredc2)
    img_fltc3_aux = np.fft.ifftshift(img_filteredc3)

    # Inverse DFT
    img_fltc1 = cv2.idft(img_fltc1_aux)
    img_fltc2 = cv2.idft(img_fltc2_aux)
    img_fltc3 = cv2.idft(img_fltc3_aux)

    # Se obtiene la magnitud de la parte real e imaginaria de cada canal en frecuencia
    img_fltc1 = cv2.magnitude(img_fltc1[:,:,0],img_fltc1[:,:,1])
    img_fltc2 = cv2.magnitude(img_fltc2[:,:,0],img_fltc2[:,:,1])
    img_fltc3 = cv2.magnitude(img_fltc3[:,:,0],img_fltc3[:,:,1])

    # Se normalizan los valores de cada canal en 0 y 255
    img_fltc1 = cv2.normalize(img_fltc1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img_fltc2 = cv2.normalize(img_fltc2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img_fltc3 = cv2.normalize(img_fltc3, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    # Se juntan los canales en una sola imagen
    img_flt = cv2.merge((img_fltc1, img_fltc2, img_fltc3))
    #cv2.imwrite('img_flt.png', img_flt)

    h,w = img_flt.shape[0:2]
    # Mientras las dimensiones no sean divisibles entre 4
    while h%4 != 0:
        img_aux = img_flt.copy()
        img_flt = img_aux[0:-1,:,:]
        h, w = img_flt.shape[0:2]
    while w%4 != 0:
        img_aux = img_flt.copy()
        img_flt = img_aux[:,0:-1,:]
        h, w = img_flt.shape[0:2]    

    img_dwn = img_flt[0::K,0::K] 

    # Se interpola la imagen sub-muestreara
    img_resize = cv2.resize(img_dwn, None, fx=K, fy=K, interpolation = cv2.INTER_CUBIC)
    # Se calcula el error entre la imagen interpolada y la imagen original
    J = 1/(len(img.flat))*(np.sum(np.abs(np.subtract(img_resize,img))))

    return J, img_dwn

'''
PSEUDO-ALGORITMO - MAIN:
1. Ubicar el directorio relativo 'X/' contenedor de todas las imagenes HR a
sub-muestrear
2. Crear los directorios internos donde se almacenaran los resultados: 
imagenes sub-muestreadas e imagenes sub-muestreadas->sobre-muestreadas bicubicamente.
3. Obtener una lista con todos los nombres de archivos de las imagenes HR, ordenarlo
en orden alfanumerico.
4. Recorrer la lista de archivos.
    1. Sub-muestrear espacialmente la imagen y almacenar los resultados 
    en su directorio correspondiente.
    
PSEUDO-ALGORITMO - sub-muestreo ESPACIAL:
Con el fin de evitar aliasing en la imagen sub-muestreada.
Para cada imagen HR en la lista de archivos:
    1. Obtener el factor de sub-muestreo y el numero de iteraciones
    como entrada del usuario.
    2. Generar los 'itr' filtros Gaussianos a probar, el primer filtro
    Gaussiano tendra una frecuencia de corte igual a la desviacion estandar: fcx = fsx/(2*K),
    fcy = fsy/(2*K), siendo 'K' el factor de sub-muestreo. La frecuencia de corte en 'x' y en 'y'
    de los siguientes filtros ira aumentando en 1 hasta generar los 'itr' filtros.
    3. Para cada filtro:
        1. Se filtrara la imagen con el filtro y se sub-muestreara una vez filtrada
        2. La imagen filtrada se interpolara (escalara o sobre-muestreara) 
        bicubicamente en el mismo factor 'K' de sub-muestreo.
        3. Se calcular el error absoluto medio entre la imagen original y la 
        imagen filtrada-sub-muestreara-interpolada
    4. El filtro con menor error sera elegido y su resultado sera almacenado
'''


print("\nEl directorio de trabajo es: " + os.getcwd())

HR_dirs = "Campos_de_luz/ajedrez"
print ("\nLos directorios que contienen cada campo de luz HR son : " + str(HR_dirs)) 

# Especificar el factor de sub-muestreo y el numero de iteraciones
Ki = input("\nPor favor, introduzca el factor de sub-muestreo, debe ser divisible entre 2 en la medida de lo posible (se tomara el numero entero del valor introducido): ")
K = int(Ki)
if (K <= 0) and (K > 10) and (K%2 != 0):
    print("Numero invalido, se elegirá el valor por defecto K = 4")
    K = 4

itri = input("Por favor, introduzca el numero de iteraciones para optimizar la sub-muestreo de cada imagen (se tomara el numero entero del valor introducido): ")
itr = int(itri)
if (itr <= 0) and (itr > 300):
    print("Numero invalido, se elegirá el valor por defecto itr = 100")
    itr = 100
 
start_time = time.time()

print("\nSe sub-muestrearan todas las imagenes en un factor %d, con %d iteraciones en cada una para optimizar la frecuencia de corte del filtro pasabajos..." % (K, itr)) 

# Se elimina los directorios si ya existen
if os.path.isdir('ajedrez_submuestreados'):
    rmdir = 'ajedrez_submuestreados'
    shutil.rmtree(rmdir)
if os.path.isdir('Filtros_Gaussianos'):
    rmdir = 'Filtros_Gaussianos'
    shutil.rmtree(rmdir)
     
# Se crean los directorios "Y", "Filtros_Gaussianos" y "Interpolacion_bicubica":
# Se crean los directorios objetivos para almacenar las imagenes sub-muestreadas espacialmente,
# los filtro Gaussianos, y las imagenes sub-muestreadas pero interpoladas en el mismo factor por una funcion bicubica.
os.makedirs((os.getcwd() + '/ajedrez_submuestreados' ), mode=0o777, exist_ok=False) 
os.makedirs((os.getcwd() + '/Filtros_Gaussianos' ), mode=0o777, exist_ok=False) 

print("Se crearon los directorios objetivos para almacenar los resultados.")

#  Se obtienen los nombres de los archivos de las imagenes de sub-apertura
filelist=os.listdir(HR_dirs)
for fichier in filelist[:]:
    if (fichier.endswith(".png") == True) or (fichier.endswith(".jpg") == True): # Remueve nombres de archivos que no sean .png
        None
    else:       
        filelist.remove(fichier)
filelist.sort()
 
print("\nArchivos a sub-muestrear:")
print(filelist)

# Se recorren las imagenes de sub-apertura encontradas
j = 0
for files in filelist:

    start_time_aux = time.time()
    
    # Imagen a sub-muestrear
    print("\nProcesando imagen: Campos_de_luz/ajedrez/" + files)
    img = cv2.imread(('Campos_de_luz/ajedrez/' + files), cv2.IMREAD_COLOR)
    
    h, w = img.shape[0:2]
    # Se verifican las dimensiones de la imagen, que sean divisibles entre 4
    if h%4 == 0 and w%4 == 0:
        print("Las dimensiones de la imagen son correctamente divisibles entre 4")
    else:
        print("Las dimensiones de la imagen no son incorrectamente divisibles entre 4, eliminando bordes extra...")
        while h%4 != 0:
            img_aux = img.copy()
            img = img_aux[0:-1,:,:]
            h, w = img.shape[0:2]
        while w%4 != 0:
            img_aux = img.copy()
            img = img_aux[:,0:-1,:]
            h, w = img.shape[0:2]    

    img_aux = borde_impar(img)

    # Frecuencia de corte inicial, teoricamente dada por el Teorema de Muestro de Nyquist
    # fcx = fsx/(2*K)
    # fcy = fsy(2*K)
    # Ambos valores teoricos se disminuyen 10 pixeles, por consideraciones de implementacion
    sigmax = int((1/(2*K))*(img_aux.shape[0]))
    sigmay = int((1/(2*K))*(img_aux.shape[1]))

    sigmasx = np.arange(sigmax, sigmax+itr, 1)
    sigmasy = np.arange(sigmay, sigmay+itr, 1)
    
    # Generacion de los filtros Gaussianos a evaluar
    gaussian = []
    gaussian = [filtro_gaussian(img_aux, sigmasx[i], sigmasy[i]) for i in range(len(sigmasx))]

    # Extraccion de los canales de color de la imagen a sub-muestrear
    b,g,r = cv2.split(img_aux)

    # FFT para cada canal de color
    img_fft_img_c1 = cv2.dft(np.float32(b),flags = cv2.DFT_COMPLEX_OUTPUT)
    sfft_img_c1 = np.fft.fftshift(img_fft_img_c1)

    img_fft_img_c2 = cv2.dft(np.float32(g),flags = cv2.DFT_COMPLEX_OUTPUT)
    sfft_img_c2 = np.fft.fftshift(img_fft_img_c2)

    img_fft_img_c3 = cv2.dft(np.float32(r),flags = cv2.DFT_COMPLEX_OUTPUT)
    sfft_img_c3 = np.fft.fftshift(img_fft_img_c3)

    # Listas donde se almacenaran los resultados
    Js = []
    img_dwn = []

    # Se empiezan a evaluar los filtros 
    for gauss in gaussian:
        j_aux, img_aux_dwn = iteraciones_opt(img, gauss, sfft_img_c1, sfft_img_c2, sfft_img_c3, K)
        Js.append(j_aux)
        img_dwn.append(img_aux_dwn)

    # Se obtiene el indice del filtro Gaussiano con menor error
    ind_min = np.argmin(Js)
    print("Con Sigmax=%d, Sigmay=%d, el costo entre la imagen original y la imagen escalada es J=%f" % (sigmasx[ind_min], sigmasy[ind_min], np.min(Js)))  
    
    # Se guarda la imagen sub-muestreara por el filtro Gaussiano con menor error
    name = 'ajedrez_submuestreados/LR_' + files
    cv2.imwrite(name, img_dwn[ind_min])
    
    # Se guarda el filtro Gaussiano
    norm_gaussian = cv2.normalize(gaussian[ind_min], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_gaussian.astype(np.uint8)
    name = os.getcwd() + '/Filtros_Gaussianos/flt_gaussian_' + files
    cv2.imwrite(name, norm_gaussian[:,:,0])
    
    j += 1
    
    print("Se han almacenado correctamente los resultados de la imagen %d..." % (j))
    print("---Tiempo de ejecucion: %s segundos ---" % (time.time() - start_time_aux))
            
print("Las %d imagenes se han sido sub-muestrearas espacialmente correctamente..." % (j))

print("--- Execution time: %s seconds ---" % (time.time() - start_time))








