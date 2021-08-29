import numpy as np
import logging
import pathlib
import os

seq_path = './sequences'
files = os.listdir(seq_path)
print(files)
image_sets_file = './train_fileNames_with_sequenceNum_forLstm.txt'
file_write_obj = open(image_sets_file,'w')

for i in files:
    pics_name = os.listdir(seq_path+'/'+i)
    print(len(pics_name))



for i in range(len(file)):
    image_id = file[i].split(',')
    #print(len(image_id))
    for j in image_id:

        file_write_obj.writelines(j)
        file_write_obj.write('\n')
        #result.append(j)
file_write_obj.close()