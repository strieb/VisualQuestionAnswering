# https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/read_tsv.py

from Environment import DATADIR
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap


csv.field_size_limit(1000000)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = DATADIR+'/Database/trainval_resnet101_faster_rcnn_genome_36.tsv'

imgDir = DATADIR+'/Images/both2014/rcnn'

if __name__ == '__main__':

    i = 0
    # Verify we can read a tsv
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.b64decode(item[field]), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            np.save(imgDir+"/"+str(item['image_id']), item['features'])
            i += 1
            if(i % 1000 == 0):
                print(i)
