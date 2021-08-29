import numpy as np
import cv2
import random
from PIL import Image
from sort import Sort
import time
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
import csv
import os
import sys
from tqdm import tqdm, trange
from bs4 import BeautifulSoup
import xlwt

# parameters used to configure the font and color for labels and boundingboxes in visulisation.
color = [(0,0,0),(125,0,125),(0,125,125),(125,125,0),(255,0,0),(0,0,255),(0,255,0)]
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
lineType               = 2

# Counter class definition
class counter():
    def __init__(self):
        self.count = 0
        self.counter = {}
        self.object_list = []
        self.frames = 0 
        self.tracker = Sort()
        self.classes = ['human','car','bicycle','truck','escooter','motorbike','bus']
        self.input_array= [] # used to reorder the input detection
        
        # initialization of counters and sessions
        for i in range(len(self.classes)):
            # counter for different kinds and overall counter
            self.counter[self.classes[i]] = []
        self.track_result = []
        
    def add_object(self, new_object):
        self.object_list.append(new_object)

    def clear_object_list(self):
        self.object_list = []

        
    def count_the_object(self):
        to_drop = []
        for index,obj in enumerate(self.object_list):
            if obj.count_flag ==True:
                self.count = self.count + 1
                to_drop.append(index)
        if len(to_drop)>0:
            off_set = 0
            for index in to_drop:
                del self.object_list[index-off_set]
                off_set = off_set + 1
				
				
    #Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    def iou(self, bb_test,bb_gt):
        
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
            + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)

    #Update trackers with detections in the current frame and match their predictions with detections to handel the tracking task.
    def tracking_counting_kalman(self):
        detections = []
        ious = [[0] * len(self.object_list) for i in range(len(self.object_list))]
        for obj in self.object_list:
            detections.append(obj.box())    
        detections = np.array(detections)
        trackers,gt_trk= self.tracker.update(detections)
        return trackers,gt_trk

    #This function is used to add id and boundingbox to the origin input picture to do the tracking/detection visulisation
    def drawTrackerOnPic(self,trackers ,seq,filePath, pic_num, target_dir = './trackDrawOnPic/'):
        
        img = Image.open(filePath+'/'+pic_num+'.pgm')
        img = np.array(img.resize((320*3, 224*3)))
        img = img.astype(np.uint8) 
        area_list = []
        for obj in self.object_list:
            area_list.append(obj.box_h*obj.box_h)
        if len(area_list)>0:
            zipped = zip(range(len(area_list)),area_list)
            zipped = sorted(zipped, key = lambda t: t[1])
            index_list = [i[0] for i in zipped]
            for index in index_list:
                obj = self.object_list[index]
                topleft_x,topleft_y,buttom_x,buttom_y = obj.box()
                temp_h = min((buttom_y-topleft_y),obj.box_h)
                temp_w = min((buttom_x-topleft_x),obj.box_w)           

        if trackers is not None:

            for d in trackers:
                d = [int(i) for i in d]
                cv2.rectangle(img,(d[0]*3,d[1]*3),(d[2]*3,d[3]*3),(255,255,220),3)
                cv2.putText(img,'ID_trackers {}'.format(d[4]),    (max(0,d[0]*3-3),max(0,d[1]*3-7)), font, fontScale,(255,255,200),lineType)
        for obj in self.object_list:
            topleft_x,topleft_y,buttom_x,buttom_y = obj.box()
            cv2.rectangle(img,(topleft_x*3,topleft_y*3),(buttom_x*3,buttom_y*3),obj.color,3)
            cv2.putText(img,'ID_groundTruth {}'.format(obj.name),    (max(0,topleft_x*3-3),max(0,topleft_y*3-7)), font, fontScale,obj.color,lineType)
        
        cv2.putText(img,'Frames {}'.format(self.frames),    (20,20), font, fontScale,(255,0,0),lineType)   
        cv2.putText(img,'Counting {}'.format(self.count),    (280,20), font, fontScale,(255,0,0),lineType)   
   
        if not os.path.exists(target_dir+seq):
            os.mkdir(target_dir+seq)

        cv2.imwrite(target_dir+seq+'/'+pic_num+'.jpg', img)
        self.frames = self.frames + 1



class ground_truth:
    def __init__(self, class_name = "bicycle", posAndShape=None, detection=None):

        self.class_name = detection[0] # in the 
        self.name = 0
        self.box_h = detection[4]-detection[2]
        self.box_w = detection[3]-detection[1]
        self.pos_x = (detection[3]+detection[1])/2
        self.pos_y = (detection[4]+detection[2])/2
        self.topleft_x = detection[1]
        self.topleft_y = detection[2]
        self.buttom_x = detection[3]
        self.buttom_y = detection[4]
        self.color = color[0]
        
        self.count_flag = False
        

    
    def box(self):       
        return (self.topleft_x,self.topleft_y,self.buttom_x,self.buttom_y)

#use for loading inference result as ground truth
#path is the root of all segmented sequence folders 
def readInference(path, annotations_dir):
    files = os.listdir(path)
    files.sort()
    detections = []
    image_nameList = []
    for j in files:
        result=[]
        if j[-7:]=='sub.pgm':
            image_nameList.append(j[:-8])
            with open(annotations_dir+'/'+j[:-8]+'.txt','r') as f:
                for line in f:
                    temp = list(line.strip('\n').split(' '))
                    result.append([temp[0],int(temp[2]),int(temp[3]),int(temp[4]),int(temp[5])])
                detections.append(result)

    return detections,image_nameList


#This use to save counting result for each segmented sequence in a xml file, 
def writeXml(result, default_path='count_result.xls'):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1')
    style = xlwt.XFStyle()  
    worksheet.write(0, 0, 'seq_name')  
    worksheet.write(0, 1, 'bicycle num')  
    worksheet.write(0, 2, 'frame rate')  
    for num, i in enumerate(result):
        worksheet.write(num+1, 0, str(i[0]))
        worksheet.write(num+1, 1, i[1])
        worksheet.write(num+1, 2, i[2])
    workbook.save(default_path)




# entrence for the counting logic
def main():
    ###################################
    # set config
    saveTrackerVisulisation = False
    saveXml = False
    inferenceResultDir = '../all_inference_test_set_modelid17/test4'#'../test4'
    threthhold=0
    addedNewdata = True
    dir_path = '../../Data_orderd_with_name_rules/All_test_segmented_sequences'
    xml_output_path='count_result_model_dynamicT=3.xls'
    output_visulisations_pic_dir = './trackDrawOnPic_t=3_dynamic/'

    ###################################


    #the next chunk will initialise the counting result parameter as well import ground truth to do the tracking
    # initialize different kinds of counting result parameter

    # we assume a sequence has a frame rate bigger than 6 to be a normal sequence, or it will be seen as a low frame sequence, this is used for statistic     
    framebiggerthan6 = 0
    whoHasLowFrame = []

    res_count = []
    lists =os.listdir(dir_path)
    lists.sort()
    sum_bicy = 0
    total_sequence_length=0
    total_tracker_number=0
    for k in lists:
        VOC_2007_annotations_dir = inferenceResultDir
        VOC_2007_trainval_image_set_filename = dir_path+'/'+str(k)
        classes = ['background','human','bicycle','truck','car','bus','escooter','motorbike']

        traffic_counter = counter()
        #use for checking current result in sequence k
        #print('This is the sequence: ',k)

        res,image_nameList = readInference(VOC_2007_trainval_image_set_filename,VOC_2007_annotations_dir)
        avg_frame_per_second = round(1/((int(image_nameList[-1][15:])-int(image_nameList[2][15:]))/(1000*(len(image_nameList)-2))),1)
        track_result = []
        count_dic = {}
        numberOfCount =0
        lastTracker =None
        trackerIncrement = 0
        thisTimeTracker = None
        for num,i in enumerate(res):
            thisTimeMinTracker = None
            for detection in i :
                temp_object = ground_truth(detection=detection)
                traffic_counter.add_object(temp_object)
            trackers,gt = traffic_counter.tracking_counting_kalman()
            trackers = sorted(trackers, key=lambda x:x[-1])
            gt = sorted(gt, key =lambda x:x[1])
            reOrderTracker = False

            if len(gt)>0:
                thisTimeMinTracker = gt[0][1]
                if gt[0][1] !=0:
                    reOrderTracker=True
                temp_gt = gt
            for j in gt:
                if reOrderTracker:                
                    j[1]= j[1]-thisTimeMinTracker
                    if j[1]>=len(trackers):
                            continue
                
                if i[j[0]][0]=='bicycle':
                    if j[1]>=len(trackers):
                        break
                    if trackers[j[1]][4] not in count_dic:
                        count_dic[trackers[j[1]][4]]=1
                    else:
                        count_dic[trackers[j[1]][4]]+=1
                    
                    #this is used to check the current count dictionary [id:tracked counts] 
                    #print('count_dic_INNER',count_dic)

            if saveTrackerVisulisation:
                 # parameter format: filePath, seq_num, pic_num
                traffic_counter.drawTrackerOnPic(trackers,k,VOC_2007_trainval_image_set_filename,image_nameList[num],output_visulisations_pic_dir)			
            traffic_counter.count_the_object()
            traffic_counter.clear_object_list()
        
        # used for checking counting dictionary in every sequence
        #print('count dic',count_dic)
        #print('length',numberOfCount)
        
        for i in count_dic:
            total_sequence_length+=count_dic[i]
            total_tracker_number+=1
            if count_dic[i]>=threthhold:
                numberOfCount+=1
                # to get the average tracker length
                

        if avg_frame_per_second>=6:
            framebiggerthan6+=1
        else:
            whoHasLowFrame.append(k[:-4])
        
        res_count.append([k,numberOfCount,avg_frame_per_second])

    if saveXml:
        writeXml(res_count,default_path=xml_output_path)
    
    # result analyse
    # this is counting sequence num
    # initialize different kinds of counting result parameter

    all_bicycle = 0
    # corresponding sequence num
    singleBicycle_no_occlu_near =0
    singleBicycle_no_occlu_middle =0
    singleBicycle_no_occlu_far  =0

    twoObjBicycle_short_occlu_near =0
    twoObjBicycle_short_occlu_middle =0
    twoObjBicycle_short_occlu_far  =0

    twoObjBicycle_long_occlu_near =0
    twoObjBicycle_long_occlu_middle =0
    twoObjBicycle_long_occlu_far  =0

    multiObjectBicycle = 0

    #this counting corresponding object num
    singleBicycle_no_occlu_near_num =0
    singleBicycle_no_occlu_middle_num =0
    singleBicycle_no_occlu_far_num  =0

    twoObjBicycle_short_occlu_near_num =0
    twoObjBicycle_short_occlu_middle_num =0
    twoObjBicycle_short_occlu_far_num  =0

    twoObjBicycle_long_occlu_near_num =0
    twoObjBicycle_long_occlu_middle_num =0
    twoObjBicycle_long_occlu_far_num  =0

    multiObjectBicycle = 0
    multiObjectBicycle_num  =0


    counterss=0
    #classification counting result based on input folders' name
    for i in res_count:
        j = i
        i = i[0]
        all_bicycle+=j[1]
        if i[-6:-2] =='0002' or i[-6:-4] == '10':
            counterss+=1
            if i[-1]=='0':
                singleBicycle_no_occlu_near+=1
                singleBicycle_no_occlu_near_num+=j[1]
            elif i[-1]=='1':
                singleBicycle_no_occlu_middle+=1
                singleBicycle_no_occlu_middle_num+=j[1]
            elif i[-1] =='2':
                singleBicycle_no_occlu_far+=1
                singleBicycle_no_occlu_far_num+=j[1]
        elif i[-4:-2] == '12' or i[-4:-2] == '13' or i[-4:-2] == '14' or i[-4:-2] == '17' or i[-4:-2] == '18':
            counterss+=1
            if i[-6:-4] =='12':
                if i[-1]=='0':
                    twoObjBicycle_short_occlu_near+=1
                    twoObjBicycle_short_occlu_near_num+=j[1]
                elif i[-1]=='1':
                    twoObjBicycle_short_occlu_middle+=1
                    twoObjBicycle_short_occlu_middle_num+=j[1]
                elif i[-1] =='2':
                    twoObjBicycle_short_occlu_far+=1
                    twoObjBicycle_short_occlu_far_num+=j[1]
            if i[-6:-4] =='13':
                if i[-1]=='0':
                    twoObjBicycle_long_occlu_near+=1
                    twoObjBicycle_long_occlu_near_num+=j[1]
                elif i[-1]=='1':
                    twoObjBicycle_long_occlu_middle+=1
                    twoObjBicycle_long_occlu_middle_num+=j[1]
                elif i[-1] =='2':
                    twoObjBicycle_long_occlu_far+=1
                    twoObjBicycle_long_occlu_far_num+=j[1]
        else:
            counterss+=1
            multiObjectBicycle+=1
            multiObjectBicycle_num+=j[1]

    # output of counting results in categories
    print('Total bicycle num: ',all_bicycle)
    print('counterssss seq num: ',counterss)
    print('singleBicycle_no_occlu_near: ',singleBicycle_no_occlu_near_num) 
    print('singleBicycle_no_occlu_middle: ',singleBicycle_no_occlu_middle_num)
    print('singleBicycle_no_occlu_far: ',singleBicycle_no_occlu_far_num)

    print('twoObjBicycle_short_occlu_near: ',twoObjBicycle_short_occlu_near_num)
    print('twoObjBicycle_short_occlu_middle: ',twoObjBicycle_short_occlu_middle_num)
    print('twoObjBicycle_short_occlu_far: ',twoObjBicycle_short_occlu_far_num)

    print('twoObjBicycle_long_occlu_near: ',twoObjBicycle_long_occlu_near_num)
    print('twoObjBicycle_long_occlu_middle: ',twoObjBicycle_long_occlu_middle_num)
    print('twoObjBicycle_long_occlu_far: ',twoObjBicycle_long_occlu_far_num)

    print('multiObjectBicycle: ',multiObjectBicycle_num)
    print('total tracked frames length', total_sequence_length)
    print('total tracker number', total_tracker_number )
    print('average_tracker_length',total_sequence_length/total_tracker_number)

		
		
if __name__ == '__main__':
    main()
    
    

    
    
    
        
            
