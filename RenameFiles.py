# Name: Breno Lívio Silva de Almeida
# NUSP: 10276675

# SCC0251 - Image Processing
# Project: Segmentation of Cell Cycles Images
# 2021/1

import os

def renameFiles():
    """
    Function for renaming files from Kaggle.
    """

    src1 = "Data/kaggle/interphase/"; src2 = "Data/kaggle/mitosis/"
    dest1 = "Data/Original/interphase/"; dest2 = "Data/Original/mitosis/"

    for count, filename in enumerate(os.listdir(src1)):
        dst = "I" + str(count) + ".jpg"
        src = src1 + filename
        dst = dest1 + dst
          
        os.rename(src, dst)

    for count, filename in enumerate(os.listdir(src2)):
        dst = "M" + str(count) + ".jpg"
        src = src2 + filename
        dst = dest2 + dst
          
        os.rename(src, dst)

def main():
    renameFiles()
  
if __name__ == '__main__':
    main()