from PIL import Image
import os
import shutil

root = './mask/'
path = os.listdir(root) 
for i in path:
	p2=root+i+'/'
	p = os.listdir(p2)
	for j in p:
		p3=root+i+'/'+j+'/'
		q = os.listdir(p3)
		for k in q:
			shutil.copy(p3+k, './'+'allImage'+'/'+k)
			

		
