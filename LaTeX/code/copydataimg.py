# copydataimg.py       made by: Ysbrand Galama, februari 2015
#
# Used to copy the images of the dataset according to the output of Annotator.java
# 

import subprocess as sp

# open all the files 
names = ["DATA001","DATA002","DATA003","DATA004","DATA005"]
filedir = [ (".\\"+s+".wmvout\\", open(s+".txt") ) for s in names]

above = ".\\DATA_AWATER\\";
below = ".\\DATA_BWATER\\";


ab = 1;
be = 1;
for dir,file in filedir:
	i = -1;
	for line in file:
		i += 1;
		# skip the first line
		if i == 0:
			continue
		# get the annotations on each line
		a,w,p,s = line[2], line[0] == '1', line[1], line[3] == '1'
		if s:
			# use annotations to copy images
			if w:
				fname = below+str(be)+"_"+a+p+".jpg"
				be += 1
			else:
				fname = above+str(ab)+"_"+a+p+".jpg"
				ab += 1
			
			# copy from shell
			command = ['COPY',dir+str(i)+".jpg",fname]
			sp.call(command,shell=True)