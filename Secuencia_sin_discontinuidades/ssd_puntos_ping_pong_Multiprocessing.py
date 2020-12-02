import multiprocessing
import cv2
import numpy as np
import os
import time
import shutil

start_time = time.time()

def get_newfilename(j, filename, folder):
    '''
    Funcion para obtener el nuevo nombre del archivo a reescribir
    Inputs:
        j: frame actual para organizar, ejemplo 000, 001, 002,...
        filename: nombre del archivo del frame a organizar
        folder: nombre del directorio del campo de luz analizado actualmente
    Outputs:
        new_filename: Nuevo nombre del archivo a reescribir
    '''
    unidad = j%10
    decena = int(j%100 / 10)
    centena = int(j%1000 / 100)

    new_filename = os.getcwd() + '/' + folder + '/' + 'img' + str(centena) + str(decena) + str(unidad) + '_' + filename
    name = 'img' + str(centena) + str(decena) + str(unidad) + '_' + filename
    return new_filename, name
    
def rename_frame(j, filename, folder):
    '''
    Funcion para renombrar el archivo especifico
    Inputs:
        j: frame actual para organizar, ejemplo 000, 001, 002,...
        filename: nombre del archivo del frame a organizar
        folder: nombre del directorio del campo de luz analizado actualmente
    '''
    new_filename, name = get_newfilename(j, filename, folder)
    #print("new filename: "+ str(new_filename))

    old_filename = os.getcwd() + '/' + folder + '/' + filename
    #print("old filename: "+str(old_filename))
    
    os.rename(old_filename, new_filename)
    return name
    
def optical_flow_cal(prev, filename, lk_params, p0):
    '''
    Funcion para calcular la magnitud total del flujo optico entre 
    los puntos encontrados en un fotograma de referencia 'prev' y 
    un nuevo fotograma correspondiente al archivo 'filename'.
    Inputs:
        prev: frame de referencia
        filename: nombre del archivo del frame a comparar con el frame de referencia
        lk_params: parametros del algoritmo Lucas-Kanade para calcular el flujo optico
        p0: esquinas en interes del frame de referencia para buscar en el frame de 'filename'
    Outputs:
        mag_total: Magnitud media del flujo optico entre ambos fotogramas
    '''
    nxt = cv2.imread(filename, 0)

    # calculate optical flow
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    dif = np.subtract(good_old, good_new)
    #xtotal = np.sqrt(np.sum(np.power(dif[:,0],2)))/len(dif)
    #ytotal = np.sqrt(np.sum(np.power(dif[:,1],2)))/len(dif)
    #xtotal = np.mean(np.abs(dif[:,0]))
    #ytotal = np.mean(np.abs(dif[:,1]))

    xtotal = np.mean(np.abs(dif[:,0]))
    ytotal = np.mean(dif[:,1])

    return xtotal, ytotal


def calculate(func, args):
    '''
    #https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing-programming
    Funcion para calcular el resultado de la tarea de cada
    trabajador (subproceso)
    Inputs:
        func: funcion a resolver por el trabajador
        args: argumentos de entrada de la funcion
    Outputs:
        result: resultado de la funcion elaborada
    '''
    result1, result2 = func(*args)
    return result1, result2   

def test(prev, filelist, lk_params, p0, HR_dirs, mean_xdif, mean_ydif):
    '''
    #https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing-programming
    Funcion para calcular el flujo optico entre el cuadro de 
    referencia 'prev' y el resto de cuadros restantes por organizar.
    Esta funcion inicializa una piscina de un numero de subprocesos
    alocados en cada nucleo disponible de la CPU.
    Inputs:
        prev: frame de referencia
        filelist: nombres de los archivos de los frames a comparar con el frame de referencia
        lk_params: parametros del algoritmo Lucas-Kanade para calcular el flujo optico
        p0: esquinas en interes del frame de referencia para buscar en el frame de 'filename'
        HR_dirs: Nombre del directorio del campo de luz actual
        mean_mags: Lista con la magnitud media para cada par de frames comparados
    Outputs:
        mean_mags: Lista actualizada con la magnitud media para cada par de frames comparados
    '''
    PROCESSES = 4 # 4 es el numero maximo de los nucleos de mi CPU
    print('\nCreando una piscina en la CPU con %d procesos:' % PROCESSES)

    with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(optical_flow_cal,(prev,(HR_dirs+'/'+i),lk_params, p0)) for i in filelist]

        results = [pool.apply_async(calculate, t) for t in TASKS]
 
        print('Obteniendo resultados ordenados usando pool.apply_async()...')
        for r in results:
            #print('\t', r.get())
            result1, result2 = r.get()
            mean_xdif.append(result1)
            mean_ydif.append(result2)

    return mean_xdif, mean_ydif


'''
PSEUDO-ALGORITMO - MAIN:
1. Pararse en el directorio a almacenar los resultados
2. Obtener una lista con todos los archivos dentro del directorio
3. Filtrar solo los archivos con iniciales "LF_"
4. Recorrer cada directorio HR
    1. Organizar las imagenes de sub-apertura de acuerdo al flujo optico
    entre fotogramas, se renombran los archivos en un orden ascendente como
    img000, img001, img002 ...

PSEUDO-ALGORITMO - FLUJO OPTICO ENTRE CUADROS:
1. Establecer un fotograma de referencia:
    2. Buscar un numero 'K' de esquinas, o caracteristicas, en la imagen de referencia
    3. Calcular el flujo optico por el metodo Lucas-Kanade entre los puntos  de interes
    del fotograma de referencia y el resto de fotogramas en la lista de nombres de archivo
    4. Identificar el archivo con menor flujo optico y renombrarlo tal que el fotograma
    de referencia sea su precedente.
    5. Eliminar el nombre del archivo ya renombrado de la lista con nombres de archivos.
    6. Establecer como fotograma de referencia a la imagen del archivo renombrado anteriormente.
    7. Repetir hasta que la lista con los nombres de archivos quede vacia
'''

# Parte inicial, obtener los directorios que contiene las imagenes de sub apertura de los campos de luz
pth = input("\nPor favor, introduzca la ruta de trabajo absoluta, esta ruta debe contener los directorios de cada campo de luz a procesar: ")
os.chdir(pth)

f_list = os.listdir() #'ls'
 
subs = 'LF'
dirs = [i for i in f_list if subs in i[0:2]] # Solo guarda los nombres que empiezan con 'LF'
subs = 'HR'
HR_dirs = [i for i in dirs if subs in i] # Solo guarda los nombres que contienen con 'HR'
print ("\nLos directorios que contienen cada campo de luz HR son : " + str(HR_dirs)) 

k = input("\nPor favor, introduzca el numero de esquinas a seguir, '0' es sin limite: ")

# Parametros para el detector de esquinas ShiTomasi
feature_params = dict( maxCorners = int(k), # maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned. 
                       qualityLevel = 0.03, # For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected. 
                       minDistance = 3, # Minimum possible Euclidean distance between the returned corners. 
                       blockSize = 9 ) # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. 

# Parametros para el algoritmo de flujo optico de Lucas-Kanade
lk_params = dict( winSize  = (25,25), # size of the search window at each pyramid level. 
                  maxLevel = 4, # 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel. 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)) # specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon. 

filas_grilla = int(input("\nPor favor, ingrese el numero de filas en la grilla de las imagenes de sub-apertura (debe ser impar): "))
columnas_grilla = int(input("Por favor, ingrese el numero de columnas en la grilla de las imagenes de sub-apertura (debe ser impar): "))

new_filas_grilla = int(input("\nPor favor, ingrese el nuevo numero de filas en la grilla de las imagenes de sub-apertura (debe ser impar): "))
new_columnas_grilla = int(input("Por favor, ingrese el nuevo numero de columnas en la grilla de las imagenes de sub-apertura(debe ser impar): "))


# Se recorre cada director contenedor de un campo de luz
for i in range(len(HR_dirs)):

    # Se obtiene una lista con los nombres de archivos de las imagenes de sub-apertura
    filelist=os.listdir(HR_dirs[i])
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")): # Remueve nombres de archivos que no sean .png
            filelist.remove(fichier)
    filelist.sort()
    
    print("\nRecorriendo el directorio : " + str(HR_dirs[i]))
    print("Se procede a organizar las imagenes de sub-apertura en una secuencia sin discontinuidades renombrando sus archivos de acuerdo al orden secuencial determinado por el flujo optico entre imagenes...")

    j = 0
    
    col = 1
    fila = 0
    # Matriz de imagenes de sub-apertura
    MSA = [[0 for x in range(columnas_grilla)] for y in range(filas_grilla)]

    # Tomar el primer frame como punto de partida
    prev = cv2.imread((HR_dirs[i] + '/' + filelist[0]), 0)
    # Renombrar el archivo del primer frame
    new_filename = rename_frame(j, filelist[0], HR_dirs[i])
    filelist.pop(0) 
    MSA[fila][col-1] = new_filename
    print("\nEl frame %d ha sido organizado correctamente..." % (j))
    
    #https://docs.opencv.org/4.4.0/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

    # Iterar hasta que la lista con el nombre de archivos quede vacia
    while  len(filelist) != 0:
        
        start_time_aux = time.time()
        mean_xdif = []
        mean_ydif = []

        # Multiprocesos para calcular el flujo optico entre el frame de referencia y los archivos restantes por organizar
        if __name__ == '__main__':
            multiprocessing.freeze_support()
            mean_xdif, mean_ydif  = test(prev, filelist, lk_params, p0, HR_dirs[i], mean_xdif, mean_ydif)
            
        
        flag = False

        ymin = mean_ydif.copy()
        ymin.sort(reverse=True)
        if col == columnas_grilla:
            ymin2 = ymin[0:columnas_grilla]
            fila += 1
            col = 0
        else:
            ymin2 = ymin[0:columnas_grilla-col]


        while flag == False:

            xind_min = np.argmin(mean_xdif)

            if mean_ydif[xind_min] in ymin2:
                ind = xind_min
                flag = True
            else:
                 mean_xdif[xind_min] = 1000000   


        # Identificar el nombre del frame con menor flujo optico respecto al frame de referencia
        j += 1

        col += 1 
    
        file_to_rename = filelist[ind]
        
        # El archivo a renombrar se toma como nuevo frame de referencia
        prev = cv2.imread((HR_dirs[i] + '/' + file_to_rename), 0)
        p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)
        
        # Se renombra el archivo del nuevo frame de referencia
        new_filename = rename_frame(j, file_to_rename, HR_dirs[i])
        MSA[fila][col-1] = new_filename
        filelist.pop(ind)
        
        print("El frame %d se organizo correctamente..." % (j))
        print("---Tiempo de ejecucion: %s segundos ---" % (time.time() - start_time_aux))
        print("Los frames restantes son %d" % (len(filelist)))
        
    print("Las %d imagenes del campo de luz han sido organizadas correctamente..." % (j))
    

    # Se elimina el directorio "Decimado_amgularmente" si ya existe
    pth2 = os.getcwd() + '/' + HR_dirs[i]
    os.chdir(pth2)
    if os.path.isdir('Decimado_angularmente'):
        rmdir = 'Decimado_angularmente'
        shutil.rmtree(rmdir)
        
    os.chdir(pth)
    new_dir = os.getcwd() + '/' + HR_dirs[i] + '/Decimado_angularmente' 
    os.makedirs(new_dir, mode=0o777, exist_ok=False) # Se crea el directorio para almacenar los frames decimados angularmente
    
    k_row = filas_grilla - new_filas_grilla
    k_col = columnas_grilla - new_columnas_grilla
    
    print("\nSe procede a realizar la decimacion angular sobre el campo de luz, el resultado sera almacenado en un nuevo directorio dentro del directorio del campo de luz.")
    if (k_row%2!=0) and (k_col%2!=0):
        print("El factor de decimacion angular no es par... Intente de nuevo con un factor par.")
    
    row_step = int(k_row/2)

    col_step = int(k_col/2)

    MSA_dwns = []
    MSA_dwns = MSA[row_step:-row_step]

    MSA_dwns2 = []
    for rowi in range(len(MSA_dwns)):
        MSA_dwns2.append(MSA_dwns[rowi][col_step:-col_step])
    
    for rowi in range(len(MSA_dwns2)):
        for coli in range(len(MSA_dwns2[0])):
            cpath = os.getcwd() + '/' + HR_dirs[i] + '/' + MSA_dwns2[rowi][coli]
            npath = os.getcwd() + '/' + HR_dirs[i] + '/Decimado_angularmente/' + MSA_dwns2[rowi][coli] 
            shutil.copyfile(cpath, npath)

    print("Se guardaron correctamente las imagenes decimadas....")        

print("\n--- Execution total time: %s seconds ---" % (time.time() - start_time))
