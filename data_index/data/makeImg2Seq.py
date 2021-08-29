import os 
#import cv2
from shutil import copyfile
import numpy as np

path = 'pic2seq'
files = os.listdir(path)
'''# 删除sub文件
print(len(files))
res = []
for i in files:
    if i[-7:-4] =='sub':
        os.remove('./'+path+'/'+i)
files = os.listdir(path)
print(len(files))
'''
begin = ['192.168.43.79_1598337917939.pgm','192.168.43.109_1598337189466.pgm','192.168.43.116_1596959441731.pgm','192.168.43.116_1596963600035.pgm','192.168.43.116_1596988572072.pgm','192.168.43.116_1596989830759.pgm','192.168.43.116_1596992400561.pgm','192.168.43.211_1596959439578.pgm','192.168.43.211_1596963610215.pgm','192.168.43.211_1596986822219.pgm','192.168.43.211_1596988574573.pgm','192.168.43.211_1596989829724.pgm','192.168.43.211_1596992455588.pgm','192.168.43.213_1604565369874.pgm','192.168.43.213_1604566218211.pgm','192.168.43.213_1604566960947.pgm','192.168.43.213_1604567628647.pgm','192.168.43.213_1604567846744.pgm','192.168.43.213_1604582073482.pgm','192.168.43.213_1604583007630.pgm','192.168.43.213_1604583927669.pgm','192.168.43.213_1604584335978.pgm','192.168.43.213_1604584800080.pgm','192.168.43.213_1605271542270.pgm','192.168.43.213_1605276949887.pgm','192.168.43.213_1605277816573.pgm']
end = ['192.168.43.79_1598338016583.pgm','192.168.43.109_1598337441491.pgm','192.168.43.116_1596959936513.pgm','192.168.43.116_1596963728142.pgm','192.168.43.116_1596989029493.pgm','192.168.43.116_1596990182291.pgm','192.168.43.116_1596992710605.pgm','192.168.43.211_1596959992094.pgm','192.168.43.211_1596963730485.pgm','192.168.43.211_1596987464619.pgm','192.168.43.211_1596989039502.pgm','192.168.43.211_1596990117865.pgm','192.168.43.211_1596992462064.pgm','192.168.43.213_1604565708850.pgm','192.168.43.213_1604566626039.pgm','192.168.43.213_1604567264425.pgm','192.168.43.213_1604567773435.pgm','192.168.43.213_1604568279499.pgm','192.168.43.213_1604582491455.pgm','192.168.43.213_1604583330685.pgm','192.168.43.213_1604584214589.pgm','192.168.43.213_1604584523566.pgm','192.168.43.213_1604584945313.pgm','192.168.43.213_1605272370132.pgm','192.168.43.213_1605277546451.pgm','192.168.43.213_1605278095260.pgm']
'''
s1i1 = '192.168.43.79_1598337917939.pgm'
s1i2 = '192.168.43.79_1598337989324.pgm'
s2i1 = '192.168.43.109_1598337245862.pgm'
s3i1 = '192.168.43.116_1596959775903.pgm'
s1i1 = cv2.imread('./'+path+'/'+s1i1,cv2.IMREAD_GRAYSCALE)
s1i2 = cv2.imread('./'+path+'/'+s1i2,cv2.IMREAD_GRAYSCALE)
s2i1 = cv2.imread('./'+path+'/'+s2i1,cv2.IMREAD_GRAYSCALE)
s3i1 = cv2.imread('./'+path+'/'+s3i1,cv2.IMREAD_GRAYSCALE)
#print(s1i1-s1i2
print('sum = '+str(sumofpic(s1i1,s1i2)))
#print(s1i1-s2i1)
print('sum = '+str(sumofpic(s1i1,s2i1)))
print('sum = '+str(sumofpic(s1i1,s3i1)))

print(len(begin))
num = 0 
start_id = 0
#os.mkdir('./sequences/'+str(num))
for ids, name in enumerate(files):
    if name == begin[num]:
        os.mkdir('./sequences/'+str(num))
    copyfile('./'+path+'/'+name, './sequences/'+str(num)+'/'+name)
    if name==end[num]:
        num+=1
        print(num)
print('done')

'''
files.sort()
num = 0
for ids, name in enumerate(files):
    if name in begin:
        print('this is count'+str(num))
        print('this is the id:'+str(ids))
        print(name) 
    copyfile('./'+path+'/'+name, './sequences/'+str(num)+'/'+name)
    if name in end:
        print('this is count'+str(num))
        print('this is the id:'+str(ids))
        print(name) 
        num+=1
    #         if name == begin[num]:
    #     print('num is :'+str(num)+' and the id is '+str(ids))
    # if name == end[num]:
    #     print('num is :'+str(num)+' and the id is '+str(ids))   
    #     num+=1 
        