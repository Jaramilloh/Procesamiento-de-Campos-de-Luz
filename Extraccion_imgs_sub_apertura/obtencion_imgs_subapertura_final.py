#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:09:01 2020

@author: felipe
"""

'''
    OBTENCION DE LOS CENTROS DE LOS MICROLENTES:    
        1. Umbralizar la imagen de calibracion del LF (la imagen gris)
        
        2. Aplicar operaciones morfologicas en la imagen umbralizada
        para retirar puntos de ruido interno en los microlentes y en los
        bordes de la imagen.
        
        3. Obtener los contornos en la imagen umbralizada procesada:
            utilizando OpenCV - los contornos se almacenan en una lista de
            Python que contiene todos los contornos en la imagen.
            Cada contorno individual es un arreglo Numpy con coordenadas (x,y)
            de los puntos delimitantes sobre el objeto, en este caso,
            cada microlente.
            
        4. Eliminar los contornos de los microlentes incompletos que no nos 
        interesan:
            Se utiliza la funcion boundingRect() de OpenCV, ya que calcula el
            ancho y alto del minimo rectangulo que puede contener cada contorno.
            
        5. Obtener los centros de cada contorno, es decir, los centros de cada 
        microlente:
            Se utiliza la funcion fitEllipse() de OpenCV, ya que su aproximacion
            eliptica es mas precisa para determinar el centro de cada contorno
            a nivel de sub-pixel.
            
        6. Organizar los contornos de acuerdo a un sistema de coordenadas para
        poder indexar cada microlente.
        
        7. Obtener una lista con los centros de todos los microlentes para
        cada uno de los 3 tipos de microlentes en el LF.
'''

import cv2
import numpy as np
'''
DESARROLLO:
    
1. Umbralizar la imagen de calibracion del LF (la imagen gris de 16-bits):
    
    Se utiliza el metodo Otsu ya que la imagen es bi-modal, es decir, el 
    histograma de sus intesidades tiene dos distribuciones, una para los grises
    oscuros, y otra para los grises claros, gracias a la geometría simétrica
    entre microlentes y sus distancias. 
    El metodo Otuso calcula el umbral ideal entre ambas distribuciones (
    microlentes-gris claro, y separacion entre microlentes-gris oscura).
'''

# Hiperparametros del sistema
file = input("Por favor, introduzca el archivo del lightfield a procesar sin acronimos, por ejemplo 'darthvader': ")
print("\nPor favor, introduzca la vista angular a adquirir, para esto, ingrese un desplazamiento en 'x' y en 'y' sobre el centroide de cada microlente para generar la vista en dicha nueva posicion el limite de distancia es +/-30 unidades tanto para 'x', como a 'y':")
row_t = int(input("x = "))
col_s = int(input("y = "))  
Kfactor = int(input("\nPor favor, ingrese el factor de interpolacion sobre la imagen sintetizada: "))

#lf_file = file[0:file.find('_',0,-1)] # Nombre del LF
lf_file = file

#lf_grid_file = file[0:file.find('.',0,-1)] # Nombre de imagen de calibracion del LF
lf_grid_file = lf_file

lf_grid = cv2.imread(lf_grid_file+'_Gray.png', cv2.IMREAD_UNCHANGED) # Imagen gris de 16-bits

# Umbralizacion binaria con el umbral dado por le metodo Otsu
print("\nUmbralizando la imagen de calibracion:")
otsu_threshold, lf_grid_umbralized = cv2.threshold(lf_grid, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
print("Umbral obtenido: ", otsu_threshold)

# La imagen  umbralizada se convierte de 16bits a 8bits
lf_grid_umbralized = (lf_grid_umbralized.copy()/256).astype('uint8')

# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img1_Umbralizada.png'), lf_grid_umbralized) 



'''
2. Aplicar operaciones morfologicas en la imagen umbralizada de 8bits para retirar 
puntos de ruido interno en los microlentes y en los bordes de la imagen:
    
    Se aplica una operacion morfolica de clausura (dilatacion -> erosion) para 
    eliminar los valores de ruido dentro de cada microlentem utilizando un kernel
    3x3.

    Si se quiere (flag=True):
        Se aplica una operacion morfologica de apertura para reducir los
        puntos de ruido sobre el borde de cada microlente, kernel 3x3
        
        Se aplica una operacion morfologica de apertura 
        para reducir los puntos de ruido sobre el marco de la imagen (bordes de l
        os microlentes incompletos), con un kernel 9x9.
'''

# Operacion morfoligca de clausura
kernel = np.ones((3,3),np.uint8)
lf_grid_umbralized_denoise = cv2.morphologyEx(lf_grid_umbralized.copy(), cv2.MORPH_CLOSE, kernel)

flag = False

if flag:
    # Operacion morfologica de apertura
    lf_grid_umbralized_denoise = cv2.morphologyEx(lf_grid_umbralized_denoise, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((9,9),np.uint8)
    # Operacion morfologica de apertura
    lf_grid_umbralized_denoise = cv2.morphologyEx(lf_grid_umbralized_denoise, cv2.MORPH_OPEN, kernel)
    
# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img2_Umbralizada_filtrada.png'), lf_grid_umbralized_denoise)

# Se crea una copia en 3 canales de la imagen ya procesada simplemente para poder imprimir resultados sobre esta
img = np.zeros((lf_grid_umbralized_denoise.shape[0], lf_grid_umbralized_denoise.shape[1], 3), np.uint8)
img = cv2.cvtColor(lf_grid_umbralized_denoise, cv2.COLOR_GRAY2BGR)
# Copias para imprimir resultados futuros en RGB sobre la imagen procesada
img2 = img.copy() 
img3 = img.copy() 
img4 = img.copy() 
img5 = img.copy() 
img6 = img.copy() 
img7 = img.copy() 

'''
3. Obtener los contornos en la imagen umbralizada procesada:
    
        utilizando OpenCV - los contornos se almacenan en una lista de
        Python que contiene todos los contornos en la imagen.
        Cada contorno individual es un arreglo Numpy con coordenadas (x,y)
        de los puntos delimitantes sobre el objeto, en este caso,
        cada microlente.
        
        Unicamente se almacenan los contornos externo (RETR_EXTERNAL). No
        interesa la jerarquia interna de cada contorno. 
        
        Algoritmos para extraer los contornos:
            cv2.CHAIN_APPROX_NONE : Almacena absolutamente todos los puntos
            de cada contorno
            cv2.CHAIN_APPROX_SIMPLE : Comprime los segmentos horizontales,
            verticales y diagonales, y almacena solo los puntos finales.
        
        Despues se organizan APROXIMADAMENTE de acuerdo a su origen en el 
        sistema de coordenadas (x,y) .
        
'''

# Se obtienen los contornos en una lista
contours,_ = cv2.findContours(lf_grid_umbralized_denoise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Se organizan los contornos de acuerdo a su origen (APROXIMACION - no todos se ordenan correctamente)
# Esta funcion ayuda un monton al computador para desarrollar el paso 6.
def get_contour_precedence(contour, cols, tolerance_factor): # https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours/39445901
    '''
        Funcion para pesar los contornos en una escala de izquierda a derecha y 
        de arriba a abajom, de acuerdo a su centro geometrico.
        Entrada:
            contour: contorno actual a pesar
            cols: numero de columnas de la imagen donde se re-indexara el contorno
            tolerance_factor: tolerancia en terminos de la distancia entre filas
            de los contornos
        Salida: el rank del contorno
    '''
    origin,_ = cv2.minEnclosingCircle(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
contours.sort(key=lambda x:get_contour_precedence(x, lf_grid_umbralized_denoise.shape[1], tolerance_factor = 10))
# NOTA: la funcion anterior no organiza TODOS los contornos de forma correcta, mas adelante se organizaran TODOS de forma correcta.


# Se dibujan los contornos sobre la imagen procesada en 3 canales
cv2.drawContours(img, contours, -1, (0,0,255), 1)

# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img3_Contornos_completos.png'), img)


'''
4. Eliminar los contornos de los microlentes incompletos que no nos 
interesan: Obtener unicamente 220 x 176 contornos (microlentes completos)
    
        Se eliminaran utilizando la funcion boundingRect(): Calcula el origen,
        ancho y alto del minimo rectangulo que puede contener el contorno.
        
        Primero, cualquier contorno con menos de 5 puntos será descartado 
        automaticamente al evaluar la funcion. Se obtiene el alto, el ancho,
        y la coordenada 'y' del origen de cada contorno
        
        Después, se calcula un ancho promedio y una altura promedio para 
        todos los contornos de microlentes.
        
        Por ultimo, se descartaran los contornos cuyos anchos y altos no se 
        encuentren cercanos a la media en cierto factor 'a'. 
        Se descartan aquellos contornos cuya altura se encuentra en el borde 
        (ultima fila de microlentes recortados).
        
        Los contornos finales se almacenan en 'temp2contours'.
'''

y_cnt =[] # Posicion 'y' del origen todos los rectangulos
w_cnt = [] # Anchos de todos los rectangulos
h_cnt = [] # Altos de todos los rectangulos

tmpcontours = [] # Contornos con mas de 5 puntos
for c in contours:
    if len(c) > 5:
        tmpcontours.append(c)
        _,y,w,h = cv2.boundingRect(c)
        y_cnt.append(y)
        w_cnt.append(w)
        h_cnt.append(h)
         
w_mean = np.mean(w_cnt) # Anchura promedia
h_mean = np.mean(h_cnt) # Altura promedia

flag = False # Bandera para repetir el bucle hasta eliminar todos los contornos indeseados
a = 1.5 # Factor de tolerancia para comparar cada contorno con las medidas promedio
w_MLA, h_MLA = 220, 176  # Dimensiones de la malla de microlentes esperada

while flag == False:
    temp2contours = [] # Contornos finales en desorden
    
    for i in range(len(tmpcontours)):
        if (w_cnt[i] > w_mean/a) and (h_cnt[i] > h_mean/a) and ((lf_grid_umbralized_denoise.shape[0]-y_cnt[i]) > h_mean):
            temp2contours.append(tmpcontours[i])   
            
    if len(temp2contours) == (w_MLA*h_MLA):
        print("Se han eliminado correctamente los contornos indeseados...")
        flag = True
    else:
        a -= 0.1 # Se reduce la tolerancia para eliminar contornos

# Se dibujan los contornos finales
cv2.drawContours(img2, temp2contours, -1, (0,0,255), 1)

# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img4_Contornos_filtrados.png'), img2)       


'''
5. Obtener los centros de cada contorno, es decir, los centros de cada microlente:
    
        Se utiliza la opcion fitEllipse() de OpenCV, ya que su aproximacion
        eliptica es mas precisa para determinar el centro de cada contorno
        a nivel de sub-pixel.
        
        La funcion fitEllipse() retorna una lista con una tupla de las coordenadas
        (x,y) del centroide, una tupla con las medidas ancho y alto (w,h) de
        la elipse, y un angulo de rotacion.
        
        Los centroides de las elipses de cada contorno 'temp2contours' se 
        almacenan en 'center_cnt'.
        Las elipses de cada contorno 'temp2contours' se almacena en 'all_ellipses'.
''' 

all_ellipses = [] # Elipses ajustadas en cada contorno final
center_cnt = [] # Centroide de cada contorno final

for c in temp2contours:
    ellipse = cv2.fitEllipse(c)
    all_ellipses.append(ellipse)
    center_cnt.append(ellipse[0])
    cv2.ellipse(img3, ellipse, (0,0,255), 1) # Se imprime acada elipse sobre su contorno
        
# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img5_minElipses_contornos.png'), img3)


'''
6. Organizar los contornos de acuerdo a un sistema de coordenadas para poder 
indexar cada microlente:
    
Por mediciones anteriores, se conocen las siguientes distancias en pixeles:
    Distancia horizontal entre centros de microlentes = aprox. 35
    Distancia vertical entre centros de microlentes = aprox. 30

Vale la pena mencionar que cada indice 'i' de los arreglos 'center_cnt', 
'all_ellipses', 'temp2contours', corresponden al mismo microlente (contorno).
    
    PSEUDO-ALGORITMO:
    Se crea una matriz 'MLA' de 220x176, en la cual, se almacenara en cada 
    posicion una tupla con los centroides de cada microlente.
    
    La matriz MLA fisica es hexagonal, es decir, los centros en 'x' entre
    filas NO son simetricos, entonces, los centroides en (x,y) entre filas
    van a estar desfasados: 
                     
        Cabe resaltar que la matriz MLA asume simetria, ya que al recortar 
        los contornos indeseados (microlentes incompletos), se obtienen la misma 
        cantidad de microlentes por cada fila (220).
        
    Se inicializa la 'pos_inicial' como el centroide del primer microlente en 
    la imagen, es decir, el microlente ubicado en la posicion 0,0 de la 
    matriz MLA. 
    
    Se inicializa la 'pos_inicial_row2' como el centroide del primer microlente 
    ubicado en la segunda fila, es decir, en la posicion 1,0 de la matriz MLA.
    
    Estas posiciones se determina iterando sobre todos los contornos, buscando 
    las minimas coordenadas correspondientes dentro de la imagen 7716x5364. 
    
    Con estas dos coordenadas, se determina la distancia en 'x' y en 'y': 'dist_x',
    'dist_y', entre los centroides de los microlentes.
    
    Se recorre un bucle para recorrer cada fila 'k' de la matriz MLA: 
    
        Dentro de este bucle, se recorren las columnas 'j' de la matriz MLA: 
    
            Dentro de estos dos bucles, se recorre otro bucle para identificar el 
            contorno (o microlente) correspondiente a la posicion MLA[k][j] de la matriz
            de microlentes:
                
            Se calcula una region limitante en la que se espera que caiga el centroide
            del microlente correspondiente, estos limites estan dados por las 
            medidas de fitEllipse() para cada microlente, y, la posicion actual 
            actual dada en 'pos_actual'.
                
            Se recorren todos los centroides almacenados en 'center_cnt' hasta
            encontrar el centroide 'i' que cae dentro de la region limitante. 
                
            Cuando se encuentra este centroide, se guarda:
                
                MLA[k][j] = center_cnt[i].
                
                (los siguientes dos por motivos practicos para graficarlos ordenados)
                finalcontours[k*j] = temp2contours[i]
                finalcenter_cnt[k*j] = center_cnt[i]
                
                (se actualiza la posicion inicial estimada de acuerdo a la posicion 
                 original del centroide encontrado)
                pos_actual = center_cnt 
                    
            Una vez almacenado lo anterior, se elimina el centroide encontrado 
            para reducir iteraciones en los siguientes contornos a ordenar:
                
                center_cnt.pop(i)
                temp2contours.pop(i) -> se elimina para guardar el siguiente contorno
                                        con el mismo indice logico de 'center_cnt'
                break

        Una vez se ordena un centroide en la fila 'k' de la matriz MLA, se pasa 
        a la siguiente columna 'j', se suma 'dist_x' a la coordenada 'x' de 
        'pos_actual' para buscar el siguiente centroide.     
              
    Una vez se ordena toda una fila en la matriz MLA, se debe pasar a la siguiente fila.
    
    Se suma 'dist_y' a la posicion 'y' de 'pos_actual' para asegurarnos que estaremos
    buscando en la siguiente fila. 
    
    Ahora, para reiniciar la posicion 'x' de 'pos_actual' se debe tener en 
    cuenta lo siguiente:
        
        Si la nueva fila 'k' es impar: se reinicia 'pos_actual' en 'x' = 22     
        Si es impar: se reinicia 'pos_actual' en 'x' = 39
    
'''

# Dimensiones de la malla de microlentes 
w_MLA, h_MLA = 220, 176

# Creacion de una matriz con las dimensiones de la malla de microlentes
MLA = [[0 for x in range(w_MLA)] for y in range(h_MLA)] 

# Posicion inicial (x,y) para empezar a iterar desde la posicion [0,0] de MLA.
pos_inicial = [lf_grid.shape[1], lf_grid.shape[0]]
for i in center_cnt:
    if (i[0] < pos_inicial[0]) and (i[1] < pos_inicial[1]):
        pos_inicial[0] = i[0]
        pos_inicial[1] = i[1]
print("El microlente [0, 0] de la matriz de microlentes, \nse encuentra en las coordenadas (x,y) de la imagen: %f, %f " % (pos_inicial[0], pos_inicial[1]))

# Posicion inicial (x,y) para empezar a iterar desde la posicion [1,0] de MLA.
# Se verifica que la coordenada en interes sea estrictamente mayor a la coordeanada 'y'
# de 'pos_inicial' mas un offset dado por la mitad de la altura promedio de microlentes.
pos_inicial_row2 = [lf_grid.shape[1], lf_grid.shape[0]]
for i in center_cnt:
    if (i[0] < pos_inicial_row2[0]) and (i[1] < pos_inicial_row2[1]) and (i[1] > (pos_inicial[1]+(h_mean/2))):
        pos_inicial_row2[0] = i[0]
        pos_inicial_row2[1] = i[1]
print("El microlente [1, 0] de la matriz de microlentes, \nse encuentra en las coordenadas (x,y) de la imagen: %f, %f" % (pos_inicial_row2[0], pos_inicial_row2[1]))

# Se determina la distancia en 'x' y en 'y' entre los centroides de microlentes
dist_x = np.rint((pos_inicial_row2[0] - pos_inicial[0])*2)
dist_y = np.rint(pos_inicial_row2[1] - pos_inicial[1])
       
# Listas para almacenar cada centro y cada contorno en el orden correcto
finalcontours = []
finalcenter_cnt = []

pos_actual = [pos_inicial[0], pos_inicial[1]]
for k in range(h_MLA): # Recorrer filas de la matriz de MLA  
 
    # Reiniciar la posicion inicial en 'x' para recorrer cada fila
    if k%2==0: # Si la fila es par.
        pos_actual[0] = pos_inicial[0]
        #print("\nAcabe de reiniciar pos_actual en %f,%f" % (pos_actual[0],pos_actual[1]))
    else: # Si la fila es impar
        pos_actual[0] = pos_inicial_row2[0]
        #print("\nAcabe de reiniciar pos_actual en %f,%f" % (pos_actual[0],pos_actual[1]))
        
    for j in range(w_MLA): # # Recorrer columnas de la matriz de MLA    
        for i in range(len(center_cnt)): # En este ciclo ya nos paramos en la posicion MLA[k][j]
            
            # Definicion de limites para buscar el centro correspondiente 
            # que coincida con el microlente esperado en MLA[k][j]
            down_bound = (pos_actual[1] + all_ellipses[i][1][1]/2) 
            up_bound = (pos_actual[1] - all_ellipses[i][1][1]/2)
            left_bound = (pos_actual[0] + all_ellipses[i][1][0]/2) 
            right_bound = (pos_actual[0] - all_ellipses[i][1][0]/2)
            
            # ¿el centro del microlente (contorno) esta dentro de los limites?
            if (left_bound > center_cnt[i][0]) and (center_cnt[i][0] > right_bound) and (down_bound > center_cnt[i][1]) and (center_cnt[i][1] > up_bound):             
                
                # Guardar el centro y el contorno del microlente correspondiente 
                # en la posicion del arreglo MLA[k][j]
                finalcenter_cnt.append(center_cnt[i]) 
                finalcontours.append(temp2contours[i])
                
                # Almacenar el centro del microlente en MLA[k][j]
                MLA[k][j] = center_cnt[i]
                
                # Actualizar posicion inicial estimada con la real
                pos_actual[1] = center_cnt[i][1]
                pos_actual[0] = center_cnt[i][0]
                
                # Eliminar el centro ya asignado para facilitar iteraciones futuras
                center_cnt.pop(i)
                # Eliminar el contorno ya guardado para poder guardar los otros correctamente
                temp2contours.pop(i)
                
                print("Microlente en [%d,%d] (fila, columna) de la mariz MLA: ordenado correctamente" % (k, j))
                break
            
        pos_actual[0] += dist_x     
    pos_actual[1] += dist_y
print("\nSe han organizado correctamente los microlentes dentro de la matriz MLA en su centroide (x,y) correspondiente en la imagen... :D")

# Se dibuja los contornos organizados
for i in range(len(MLA)):
    for j in range(len(MLA[0])):
        x = MLA[i][j][0]
        y = MLA[i][j][1]
        img4 = cv2.putText(img4, (str(i)+","+str(j)), (int(x-13),int(y+4)), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,20,50))
cv2.drawContours(img4, finalcontours, -1, (255,0,0), 1)
# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img6_Contornos_organizados.png'), img4)

'''
Para recapitular: Se obtenieron los centros de cada microlente y se separaron
en 3 tipos de microlente:
    
    MLA es una matriz de tuplas que almacena el centroide de cada microlente en 
    su respectivo orden dentro de la grilla de calibracion.
    
    MLA_tipo1, MLA_tipo2, MLA_tipo3 son matrices de tuplas que almacenan los
    centroides (x,y) de cada MLA por su tipo correspondiente. 
    

7. Obtener una lista con los centros de todos los microlentes para cada uno 
de los 3 tipos de microlentes en el LF:
    
    Los tipos de microlentes están intercalados por cada fila. Para las filas
    pares de la matriz MLA, el primer microlente es Tipo1. Para las filas
    impares, el primer microlente es Tipo3

OBTENCION DE LAS IMAGENES SINTETIZADAS PARA CADA TIPO DE MICROLENTE Y
PARA LOS 3 TIPOS DE MICROLENTES JUNTOS.
'''

print("\nSe extraen los centroides para cada tipo de microlente en una lista...")
# Listas para almacenar las posiciones de la matriz MLA correspondientes a cada tipo de microlentes
MLA_tipo1 = []
MLA_tipo2 = []
MLA_tipo3 = []

for i in range(len(MLA)): # Recorrer filas de la matriz de MLA               
    if i%2==0: # Si la fila es par
        a = 0 # Variable auxiliar para contar tipos de microlentes     
    else: # Si la fila es impar
        a = 2
    
    temp1 = []
    temp2 = []
    temp3 = []      
    for j in range(len(MLA[0])): # Recorrer columnas de la matriz de MLA        
        if a==0:
            temp1.append(MLA[i][j][:])
            a += 1
        elif a==1:
            temp2.append(MLA[i][j][:])
            a += 1
        elif a==2:
            temp3.append(MLA[i][j][:])
            a = 0 
   
    MLA_tipo1.append(temp1) 
    MLA_tipo2.append(temp2)
    MLA_tipo3.append(temp3)         


print("\nSe procedera a guardar las imagenes sintetizadas de la imagen cruda, ya demosaciada y balanceada en blancos...")
print("Para la distribucion de cada tipo (3 en total) de microlentes:")

lf_procss_file =  lf_file + "_Processed" # LF procesado
lf_procss = cv2.imread(lf_procss_file+'.png', cv2.IMREAD_COLOR) # Imagen sRGB de 8-bits

# Se dibujan los microlentes TIPO1, TIPO2 y TIPO3 sobre la imagen umbralizada
for i in range(len(MLA_tipo1)):
    for j in range(min([len(MLA_tipo1[0]),len(MLA_tipo1[1])])):
        x = MLA_tipo1[i][j][0]
        y = MLA_tipo1[i][j][1]
        img5 = cv2.putText(img5, str(1), (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 0.5, (50,20,255))
for i in range(len(MLA_tipo2)):
    for j in range(min([len(MLA_tipo2[0]),len(MLA_tipo2[1])])):
        x = MLA_tipo2[i][j][0]
        y = MLA_tipo2[i][j][1]
        img5 = cv2.putText(img5, str(2), (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,20,50))    
for i in range(len(MLA_tipo3)):
    for j in range(min([len(MLA_tipo3[0]),len(MLA_tipo3[1])])):
        x = MLA_tipo3[i][j][0]
        y = MLA_tipo3[i][j][1]
        img5 = cv2.putText(img5, str(3), (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 0.5, (20,255,20))
# Se guarda el resultado
cv2.imwrite((lf_grid_file + '_img7_TipoContornos1-2-3.png'), img5) 
        

# Se sintetiza la imagen central de los microlentes TIPO 1
mla_tipo1 = np.zeros((len(MLA_tipo1), min([len(MLA_tipo1[0]),len(MLA_tipo1[1])]), 3), np.uint8)
for i in range(mla_tipo1.shape[0]): # Bucle para recorrer filas
    for j in range(mla_tipo1.shape[1]): # Bucle para recorrer columnas
        x = int(np.rint(MLA_tipo1[i][j][0])) # Coordenada en X del valor de luminancia a copiar
        y = int(np.rint(MLA_tipo1[i][j][1])) # Coordenada en Y del valor de luminancia a copiar
        mla_tipo1[i,j,:] = lf_procss[y,x,:]
# Se guarda el resultado
cv2.imwrite((lf_procss_file + '_img8_VistaCentral_MLA_Tipo1.png'), mla_tipo1)
print("\nSe ha guardado la imagen central de los microlentes TIPO1....")
# Se sintetiza la imagen central de los microlentes TIPO 2
mla_tipo2 = np.zeros((len(MLA_tipo2), min([len(MLA_tipo2[0]),len(MLA_tipo2[1])]), 3), np.uint8)
for i in range(mla_tipo2.shape[0]): # Bucle para recorrer filas
    for j in range(mla_tipo2.shape[1]): # Bucle para recorrer columnas
        x = int(np.rint(MLA_tipo2[i][j][0])) # Coordenada en X del valor de luminancia a copiar
        y = int(np.rint(MLA_tipo2[i][j][1])) # Coordenada en Y del valor de luminancia a copiar
        mla_tipo2[i,j,:] = lf_procss[y,x,:]
# Se guarda el resultado
cv2.imwrite((lf_procss_file + '_img9_VistaCentral_MLA_Tipo2.png'), mla_tipo2)
print("Se ha guardado la imagen central de los microlentes TIPO2....")
# Se sintetiza la imagen central de los microlentes TIPO 3
mla_tipo3 = np.zeros((len(MLA_tipo3), min([len(MLA_tipo3[0]),len(MLA_tipo3[1])]), 3), np.uint8)
for i in range(mla_tipo3.shape[0]): # Bucle para recorrer filas
    for j in range(mla_tipo3.shape[1]): # Bucle para recorrer columnas
        x = int(np.rint(MLA_tipo3[i][j][0])) # Coordenada en X del valor de luminancia a copiar
        y = int(np.rint(MLA_tipo3[i][j][1])) # Coordenada en Y del valor de luminancia a copiar   
        mla_tipo3[i,j,:] = lf_procss[y,x,:]
# Se guarda el resultado
cv2.imwrite((lf_procss_file + '_img10_VistaCentral_MLA_Tipo3.png'), mla_tipo3)
print("Ya se ha guardado la imagen central de los microlentes TIPO3....")



print("\nPara la vista central de la matriz MLA:")
# Se sintetiza la imagen central de todos los microlentes (utilizando solo el pixel central de cada microlente)
mla_todos = np.zeros((len(MLA), len(MLA[0]), 3), np.uint8)
for i in range(mla_todos.shape[0]): # Bucle para recorrer filas
    for j in range(mla_todos.shape[1]): # Bucle para recorrer columnas
        x = int(np.rint(MLA[i][j][0])) + col_s # Coordenada en X del valor de luminancia a copiar
        y = int(np.rint(MLA[i][j][1])) + row_t # Coordenada en Y del valor de luminancia a copiar 
        
        # Si la fila de microlentes es impar, se debe promediar (o interpolar) el pixel
        # con el valor del microlente del lado izquierda, dada la geometria de la malla
        if i%2==0:
            value = lf_procss[y,x,:]                
        else: # La fila es impar
            if j==0: # Si es la primera columna, se coge el mismo valor
                value = lf_procss[y,x,:]                         
            else:
                x2 = int(np.rint(MLA[i][j-1][0])) # Coordenada en X del valor de luminancia a copiar en el microlente de la izq en caso de fila impar
                y2 = int(np.rint(MLA[i][j-1][1])) # Coordenada en Y del valor de luminancia a copiar en el microlente de la izq en caso de fila impar
         
                valuetemp = np.float64(lf_procss[y,x,:])
                valuetemp2 = np.float64(lf_procss[y2,x2,:])
                # # Interpolacion del punto intermedio
                # m = valuetemp - valuetemp2
                # value = np.uint8(valuetemp - m*0.5)
                value = np.uint8((valuetemp+valuetemp2)/2) # Promediando
                
        mla_todos[i,j,:] = value               
cv2.imwrite((lf_procss_file + '_img11_VistaCentral_Completa1x1.png'), mla_todos)
print("Se ha guardado la imagen sintetizada, utilizando un solo pixel por microlente...")


# Interpolacion de la vista central
img_central = cv2.resize(mla_todos, None, fx=Kfactor, fy=Kfactor, interpolation=cv2.INTER_CUBIC)
cv2.imwrite((lf_procss_file + '_img12_vista_interpolada.png'), mla_todos)
print("Se ha guardado la imagen sintetizada, interpolada en un factor x%d..." % (Kfactor))


'''
# Sintetizar imagen de tamaño NxN de cada vista central
sizes = [3, 5, 7, 9, 11, 13]

for s in sizes:
    w_pixels = s # Tamaño del kernel a extraer por cada microlente
    h_pixels = s
    print("\nAhora, se procede a extraer una vista central de toda la matriz MLA, pero de tamaño "+str(w_pixels)+"x"+str(h_pixels)+" para cada microlente....")
    mla_todosNxN= np.zeros((len(MLA)*h_pixels, len(MLA[0])*w_pixels, 3), np.uint8)
    
    mov_y = int(h_pixels/2) # Pasos para estar en la posicion inicial en cada microlente
    mov_x = int(w_pixels/2) 
    
    # BUCLE PARA RECORRER LA MATRIZ MLA EXTRAYENDO PIXELES
    for i in range(mla_todos.shape[0]): # Bucle para recorrer filas
        im = i*h_pixels
        for j in range(mla_todos.shape[1]): # Bucle para recorrer columnas
            jm = j*w_pixels
            x = int(np.rint(MLA[i][j][0])) - mov_x # Coordenada en X del 0,0 del kernel a recorrer
            y = int(np.rint(MLA[i][j][1])) - mov_y # Coordenada en Y del 0,0 del kernel a recorrer
    
            x2 = int(np.rint(MLA[i][j-1][0])) - mov_x # Coordenada en X del 0,0 del kernel a recorrer
            y2 = int(np.rint(MLA[i][j-1][1])) - mov_y # Coordenada en Y del 0,0 del kernel a recorrer
            
            # BUCLE PARA RECORRER EL KERNEL A EXTRAER EN CADA MICROLENTE
            for ik in range(h_pixels):
                for jk in range(w_pixels):
                    # Si la fila de microlentes es impar, se debe promediar (o interpolar) el pixel
                    # con el valor del microlente del lado izquierda, dada la geometria de la malla
                    if i%2==0:
                        value = lf_procss[(y+ik),(x+jk),:]                
                    else: # La fila es impar
                        if j==0: # Si es la primera columna, se coge el mismo valor
                            value = lf_procss[(y+ik),(x+jk),:]                         
                        else:
                            valuetemp = np.float64(lf_procss[(y+ik),(x+jk),:]) # Valor RGB del microlente actual
                            valuetemp2 = np.float64(lf_procss[(y2+ik),(x2+jk),:]) # Valor RGB del microlente de la izquierda
                            # Interpolacion del punto intermedio
                            #m = valuetemp - valuetemp2
                            #value = np.uint8(valuetemp - m*0.5)
                            value = np.uint8((valuetemp+valuetemp2)/2) # Promedio entre ambos 
                    #value = lf_procss[(y+ik),(x+jk),:]                  
                    mla_todosNxN[im+ik,jm+jk,:] = value               
    cv2.imwrite((lf_procss_file + '_img12_VistaCentral_Completa'+str(w_pixels)+'x'+str(h_pixels)+'.png'), mla_todosNxN)
    print("Se ha guardado la imagen central de todos los microlentes, utilizando "+str(w_pixels)+"x"+str(h_pixels)+" pixeles por microlente....")
            
'''




