#!/usr/bin/python3
"""Script for creating text file containing sequences of 10 frames of particular video. Here we neglect all the frames where 
there is no object in it as it was done in the official implementation in tensorflow.
Global Variables
----------------
dirs : containing list of all the training dataset folders
dirs_val : containing path to val folder of dataset
dirs_test : containing path to test folder of dataset
"""
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os

# turn images in sequences dirs to a list file for training

dirs = ['sequences/']
	


file_write_obj = open('train_fileNames_with_sequenceNum_forLstm.txt','w')
for dir in dirs:
	seqs = os.listdir(os.path.join('./'+dir))
	seq_int = []
	for i in range(len(seqs)):
		seq_int.append(i)
	seqs = seq_int

	#print(np.sort(int(seqs))
	for seq in seqs:
		seq_path = os.path.join('./',dir,str(seq))
		image_list = np.sort(os.listdir(seq_path)) 
		count = 0
		filtered_image_list = []
		for image in image_list:
			image_id = image[:-4]
            #print(image_id)
			anno_file = str(image_id) + '.xml'
            #print(anno_file)
			anno_path = os.path.join('./label2seq/',anno_file)
			objects = ET.parse(anno_path).findall("object")
			num_objs = len(objects)
			if num_objs == 0: # discarding images without object
				continue
			else:
				count = count + 1
				filtered_image_list.append(image_id)
		for i in range(0,int(count/10)):
			seq = str(seq)
			for j in range(0,10):
				if len(seq) ==1:
					seq = '0'+seq
				seqs =seq+'_'+filtered_image_list[10*i + j]
				file_write_obj.writelines(seqs)
				file_write_obj.write('\n')
file_write_obj.close()


###########################################
#transform sequence line 2 image line

'''
def _read_image_seq_ids(image_sets_file):


    seq_list = []
    with open(image_sets_file) as f:
        for line in f:
            seq_list.append(line.rstrip())
    return seq_list



image_sets_file = './train_seqs_list1.txt'
file = _read_image_seq_ids(image_sets_file)
file_write_obj = open('train_seqs_list_forBottleneckLSTM.txt','w')

print('The batch number is : %d' %len(file))
result= []
for i in range(len(file)):
    image_id = file[i].split(',')
    #print(len(image_id))
    for j in image_id:

        file_write_obj.writelines(j)
        file_write_obj.write('\n')
        #result.append(j)
file_write_obj.close()

#print(len(result))
'''