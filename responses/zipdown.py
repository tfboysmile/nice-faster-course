#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pwd')





import numpy as np
import os, pickle, zipfile, glob
from glob import glob 


zip_path = '/home/girl/Decetion/synthetic_dataset.zip'
download_dir = '/home/girl/Decetion/Data'

''' 
dst是destination的缩写，表目的
src是source的缩写，表源

'''
def zip_free(zip_path, download_dir):
    if zip_path.endswith(".zip"):
                # Unpack the zip-file.
        with zipfile.ZipFile(file=zip_path, mode="r") as zf:
            zf.extractall(download_dir)#zipfile.ZipFile(file=zip_path, mode="r").extractall(download_dir)
            print("ZIP file is extractall Done.")
    elif file_path.endswith((".tar.gz", ".tgz")):
        # Unpack the tar-ball.
         with tarfile.open(name=file_path, mode="r:gz") as ta:
                ta.extractall(download_dir)
                print(" Tarfile Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")
#zip_free(zip_path, download_dir)

