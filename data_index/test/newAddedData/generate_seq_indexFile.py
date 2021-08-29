import numpy as np
import logging
import pathlib
import os

seq_path = './mask'
files = os.listdir(seq_path)
print(files)
image_sets_file = './newAddedData.txt'
file_write_obj = open(image_sets_file,'w')

for i in files:
    print(i)
    pics_name = os.listdir(seq_path+'/'+str(i)+'/')
    for j in pics_name:
        print(j)
        pic = os.listdir(seq_path+'/'+str(i)+'/'+j)

        print(pic)



        for k in range(len(pic)):
            #image_id = pic[i].split(',')
            #print(len(image_id))
            

            file_write_obj.writelines(pic[k][:-8])
            file_write_obj.write('\n')
                #result.append(j)
file_write_obj.close()
