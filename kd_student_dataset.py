"""
Methods to read file names and corresponding labels for k-fold cross-validation
"""

import os #for creating and removing directories
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import csv
import matplotlib.pyplot as plt
import random

def my_loader(path):
  try:
    with open(path, 'rb') as f:
      img = cv2.imread(path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Image.fromarray(img)
      return img#Image.open(f).convert('RGB')
  except IOError:
      print('Cannot load image ' + path)

def get_class(label):
  classes = {
      0: 'Surprise',
      1: 'Fear',
      2: 'Disgust',
      3: 'Happiness',
      4: 'Sadness',
      5: 'Anger',
      6: 'Neutral'
  }
  return classes[label]

def get_folder(i):
  classes = {
      0 : 'MSD-E',
      1 : 'MSD-ME'
  }
  return classes[i]


class ImageList(data.Dataset):
  def __init__(self, root, subroot, fileList, train = True, transform = None, loader = my_loader):
    #subroot should be a list
    self.root = root
    self.num_cls = 7
    self.transform = transform
    self.fileList = fileList
    self.train = train
    if train == True:
      self.thresh = 0.5
    else:
      self.thresh = 0

    self.needed = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                            ])
    """
    fileL = pd.read_csv((fileList), dtype='str', header = None)
    images= fileL.iloc[:, 0].values
    """
    self.loader = loader
    dir1 = []
    dir2 = []
    labelList = []
    i = 0
    #print(root)
    for f in fileList:
      path1 = root + subroot[0]
      #print(path1)
      path2 = root + subroot[1]
      path1 = os.path.join(path1, f)
      path2 = os.path.join(path2, f)
      #print(path2)
      dir1.append(path1)
      dir2.append(path2)
      l = f.split('_')[1]
      labelList.append(int(l))
    self.labelList = labelList
    self.list1 = dir1
    self.list2 = dir2

  def __getitem__(self, index):
    label = self.labelList[index]
    imgPath1 = self.list1[index]
    imgPath2 = self.list2[index]
    #print(imgPath)
    img1 = self.loader(imgPath1)
    img2 = self.loader(imgPath2)
    if self.transform is not None and self.train is not True:
      #print("HERE")
      img1 = self.transform(img1)
      img2 = self.transform(img2)
    elif self.transform is not None and self.train is True:
      #print("here")
      transformed = self.transform(image = img1, image1 = img2)
      img1 = transformed['image']
      img2 = transformed['image1']
      img1 = self.needed(img1)
      img2 = self.needed(img2)

    r = random.uniform(0,1)
    if r < self.thresh:
      return img1, img1, label, imgPath1, imgPath2
    else:
      return img1, img2, label, imgPath1, imgPath2
  
  def __len__(self):
    return len(self.list1)

  def change_thresh(self, t):
    self.thresh = t

  
  def show_img(self, index):
    uimg, mimg, label, p1, p2 = self.__getitem__(index)
    uimg = uimg.permute(1, 2, 0)
    mimg = mimg.permute(1, 2, 0)

    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(np.rot90(uimg,0))
    f.add_subplot(1,2, 2)
    plt.imshow(np.rot90(mimg,0))
    plt.show(block=True)

    print("Label: ", label)

def get_train_val_lists(images, K, k):
  """
  Images --> list of image names
  K --> total number of folds
  k --> current fold
  Returns list of image names for val, test, and train set
  based on kth fold which can be used to get
  objects of ImageList class again
  """
  Total_IDs = 142
  #num_img_pfold = int(len(images) / K)
  num_pfold = int(Total_IDs / K)
  front = num_pfold*k
  end = num_pfold*(k+1)

  train_imgList = []
  val_imgList = []
  imgList = images
  
  val_indices = [i for i, f in enumerate(images) if int(f.split('/')[-1].split('_')[0]) in range(front, end + 1)]
  val_imgList = np.take(images, val_indices)

  train_indices = [i for i, f in enumerate(images) if int(f.split('/')[-1].split('_')[0]) not in range(front, end + 1)]
  train_imgList = np.take(images, train_indices)

  print("Number of training images: ", len(train_imgList))
  print("Number of validation images: ", len(val_imgList))
  return train_imgList, val_imgList
