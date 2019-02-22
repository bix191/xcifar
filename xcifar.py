#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def saveimage(dict,meta,dir):
    for dict_i,dict_fname in enumerate(dict[b"filenames"]):
        fname=dict_fname.decode()
        cat = meta[b"label_names"][dict[b"labels"][dict_i]].decode()
        data = dict[b"data"][dict_i]
        data = data.reshape(3,32,32)
        data = np.swapaxes(data,0,2)
        data = np.swapaxes(data,0,1)
        with Image.fromarray(data) as img:
            img.save(dir+cat+"/"+fname)

cifardir=sys.argv[1]
destdir=sys.argv[2]

dm=unpickle(cifardir+"/batches.meta")
d1=unpickle(cifardir+"/data_batch_1")
d2=unpickle(cifardir+"/data_batch_2")
d3=unpickle(cifardir+"/data_batch_3")
d4=unpickle(cifardir+"/data_batch_4")
d5=unpickle(cifardir+"/data_batch_5")
dt=unpickle(cifardir+"/test_batch")

os.makedirs(destdir)
for dir in dm[b"label_names"]:
    os.makedirs(destdir+"/training/"+dir.decode())
    os.makedirs(destdir+"/test/"+dir.decode())
    

saveimage(d1,dm,destdir+"/training/")
saveimage(d2,dm,destdir+"/training/")
saveimage(d3,dm,destdir+"/training/")
saveimage(d4,dm,destdir+"/training/")
saveimage(d5,dm,destdir+"/training/")
saveimage(dt,dm,destdir+"/test/")
