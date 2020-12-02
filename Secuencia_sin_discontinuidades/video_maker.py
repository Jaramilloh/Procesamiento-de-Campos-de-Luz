# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:58:46 2020

@author: felip
"""
import numpy as np
import cv2
import os
import os
import time


start_time = time.time()

# Funcion para obtener los frames como imagenes ya reescaladas conservando la relación de aspecto
def save_frames(fls, frame_array, folder):
    
    for i in range(len(fls)):
        #reading each files
        filename = folder + '/' + fls[i]
        img = cv2.imread(filename)
        height, width = img.shape[0:2]

        #inserting the frames into an image array
        frame_array.append(img)
    
    return frame_array, height, width
    
#Funcion para escribir el video y guardarlo como .avi
def write_video(frame_array, name, height, width):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(name,fourcc, 5.0, (width,height))
    #out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, ())

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

    out.release()



pth = input("\nPor favor, introduzca la ruta de trabajo absoluta: ")
os.chdir(pth)

f_list = os.listdir()
 
subs = 'LF'
dirs = [i for i in f_list if subs in i[0:2]] # Solo guarda los nombres que empiezan con 'LF'
subs = 'HR'
HR_dirs = [i for i in dirs if subs in i] # Solo guarda los nombres que contienen con 'HR'
print ("\nLos directorios que contienen cada campo de luz HR son : " + str(HR_dirs)) 


for i in range(len(HR_dirs)):

    start_aux_time = time.time()

    filelist=os.listdir(HR_dirs[i])
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")): # Remueve nombres de archivos que no sean .png
            filelist.remove(fichier)
    filelist.sort()
    print("\nRecorriendo el directorio : " + str(HR_dirs[i]))

    # Lista donde se guardarán los frames reescalizados
    frame_array = []
    frame_array, height, width = save_frames(filelist, frame_array, HR_dirs[i])
    name = 'video_'+ str(HR_dirs[i]) + '_completo.avi'
    write_video(frame_array, name, height, width)
    print("Se ha guardado exitosamente el video completo del campo de luz %d" % (i))

    filelist=os.listdir(HR_dirs[i]+'/Decimado_angular')
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")): # Remueve nombres de archivos que no sean .png
            filelist.remove(fichier)
    filelist.sort()
    print("\nRecorriendo el directorio : " + str(HR_dirs[i]))

    # Lista donde se guardarán los frames reescalizados
    frame_array = []
    frame_array, height, width = save_frames(filelist, frame_array, HR_dirs[i])
    name = 'video_'+ str(HR_dirs[i]) + '_decimado_angular.avi'
    write_video(frame_array, name, height, width)
    print("Se ha guardado exitosamente el video del campo de luz %d decimado angularmente" % (i))

    print("--- Execution time: %s seconds ---" % (time.time() - start_aux_time))

print("--- Execution time: %s seconds ---" % (time.time() - start_time))
