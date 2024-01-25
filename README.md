# BC_Malignancy_Investigation

In this repository we will illustrate our thesis work entitled "Deep Learning Classification and Object Detection for Breast Cancer Malignancy Investigation"


## Abstract

Breast cancer is the most commonly diagnosed cancer in women. To increase the chances of survival it is of paramount importance to detect the tumor presence in a timely manner, and this can be achieved by
means of mammography screening. The objective of the work is to identify narrow portions of the image that could potentially contain a tumor lesion (Object Detection) and to perform Image Classification only on those portions. To do this, an Object Detection algorithm was initially trained, and the areas identified by this system (called Regions of Interest or briefly RoIs), after being manually evaluated, were used to train an Image Classification algorithm whose task is to refine the Object Detection evaluation to improve the overall classification performances.

# Object Detection

## Literature

In this body of work the working process was based on the flowchart suggested by the following article ["End-to-end breast cancer detection with Python"](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftowardsdatascience.com%2Fend-to-end-breast-cancer-detection-in-python-part-1-13a1695d455)

## Data

The dataset used to perform analyses is called INbreast and is available on  [Kaggle](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset)

Inside the folder containing images, there are XML files which were created using Osirix software, and are readable only with the Pro version of the same software, which requires a monthly subscription.

On the other hand, the article followed as a guide in this project contains guidelines that allow us to read images and XML files and to properly visualize Region of Interest (ROI) with the respective coordinates that are used to build bounding boxes. Coordinates were then exported in txt format files in order for them to be read by YOLO.

For the sake of simplicity and for time reasons, we focused on only one kind of region of interest, which was MASS.

## Importing libraries

```
!pip install pydicom
%pip install -U dicomsdl

import numpy as np
import cv2
import glob
import pydicom as dicom
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import os
import albumentations as A
import plistlib
from skimage.draw import polygon
import dicomsdl
from PIL import Image
import matplotlib.pylab as plt
```

## Functions for Image preprocessing

The same article provides different functions which allow to process images and visualize bounding boxes from ROIs.

These functions can be found in the following GitHub [repository](https://github.com/jordanvaneetveldt/breast_mass_detection/blob/main/create_dataset.py), which was created from article's authors.

Of specific interest, the function *load_inbreast_mask* lets the user define the specific ROI category and produce in output the desired coordinates. In our case the argument of interest was 'MASS'.

Image preprocessing consists of four steps: cropping, normalization, image enrichment enhancement and image synthesization.

* Cropping: Considering that a big portion of the image is the background,which does not provide useful information, this will be cropped out and will be considered only the region containing breast radiography (?). The aim of this step is to retain only useful regions of the original image and to delete as many pixels as possible, as the computational effort will highly depend on that.

* Normalization: Even after cropping, the image is still composed of black pixels, which might have a negative effect on the detection process as breasts would appear less intensely. So, the breast's pixels distribution is normalized.

* Image enhancement: To enlighten the contrast between breast and the region containing cancer, the algorithm  *Contrast Limited Adaptive Histogram Normalization (CLAHE)*.

* Image synthesization: In conclusion, starting from the normalized image, a coloured one is synthesized from the two images obtained from CLAHE application with the two different thresholds, respectively equal to 1 and 2. To do this, the *merge* option, contained in the OpenCV library, is used and a coloured image is produced. This one is called a multicanal image as it is obtained from the two single ones.

## Mask

```
def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where each mass has a different number id.
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x
    i =  0
    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            if roi['Name'] == 'Mass':
                i+=1
                points = roi['Point_px']
                assert numPoints == len(points)
                points = [load_point(point) for point in points]
                if len(points) <= 2:
                    for point in points:
                        mask[int(point[0]), int(point[1])] = i
                else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = i
    return mask

def mask_to_yolo(mass_mask):
    """
    Convert a mask into albumentations format.
    @mass_mask : numpy array mask where each pixel correspond to a lesion (one pixel id per lesion)
    return: a list of list containing masses bounding boxes in YOLO coordinates:
            <x> = <absolute_x> / <image_width>
            <y> = <absolute_y> / <image_height>
            <height> = <absolute_height> / <image_height>
            <width> = <absolute_width> / <image_width>
    """
    res = []
    height, width = mass_mask.shape
    nbr_mass = len(np.unique(mass_mask))-1

    for i in range(nbr_mass):
        mask = mass_mask.copy()
        mask[mass_mask!=i+1]=0
        #find contours of each mass
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #create a bbox around the contours
        x, y, w, h = cv2.boundingRect(cnts[0])
        #convert to yolo coordinate system
        x = x+w//2 -1
        y= y+h//2 -1
        res.append([x/width,y/height,w/width,h/height, 'mass'])
    return res

#practical example of function's usage
a=load_inbreast_mask('/content/drive/MyDrive/esperimenti/esperimenti/archive/INbreast/AllXML/22580192.xml')
b=mask_to_yolo(a)

### bounding boxes
def bbox_to_txt(bboxes):
    """
    Convert a list of bbox into a string in YOLO format (to write a file).
    @bboxes : numpy array of bounding boxes
    return : a string for each object in new line: <object-class> <x> <y> <width> <height>
    """
    txt=''
    for l in bboxes:
        l = [str(x) for x in l[:4]]
        l = ' '.join(l)
        txt += '0 ' + l + '\n'
    return txt

#practical example of function's usage
bb=bbox_to_txt(b)
bb

```

## Treat Images

```
def crop(img, mask):
    """
    Crop breast ROI from image.
    @img : numpy array image
    @mask : numpy array mask of the lesions
    return: numpy array of the ROI extracted for the image,
            numpy array of the ROI extracted for the breast mask,
            numpy array of the ROI extracted for the masses mask
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w], breast_mask[y:y+h, x:x+w], mask[y:y+h, x:x+w]

#practical example of function's usage
imm2, breast , mask =crop(imm,a)

def truncation_normalization(img, mask):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    @mask : numpy array mask of the breast
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[mask!=0], 5)
    Pmax = np.percentile(img[mask!=0], 99)
    truncated = np.clip(img,Pmin, Pmax)
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[mask==0]=0
    return normalized

def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img*255, dtype=np.uint8))
    return cl

def synthetized_images(patient_id):
    """
    Create a 3-channel image composed of the truncated and normalized image,
    the contrast enhanced image with clip limit 1,
    and the contrast enhanced image with clip limit 2
    @patient_id : patient id to recover image and mask in the dataset
    return: numpy array of the breast region, numpy array of the synthetized images, numpy array of the masses mask
    """
    image_path = glob.glob(os.path.join(DCM_PATH,str(patient_id)+'*.dcm'))[0]
    mass_mask = load_inbreast_mask(os.path.join(XML_PATH,str(patient_id)+'.xml'))
    ds = dicom.dcmread(image_path)
    pixel_array_numpy = ds.pixel_array

    breast, mask, mass_mask = crop(pixel_array_numpy, mass_mask)
    normalized = truncation_normalization(breast, mask)

    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)

    synthetized = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    return breast, synthetized, mass_mask

#practical example of function's usage
breast, synthetized, mass_mask = synthetized_images('20586986')

#plotting results
plt.imshow(synthetized, interpolation='nearest')
plt.imshow(mass_mask, cmap='jet', alpha=0.5)      #comment to plot only image without its mask
plt.show()
```

##### **Notes about images without XML file**

To perform this kind of analysis, each image has to have an associated XML file. On the other hand, 65 images do not have an associated XML file, because there are no MASS regions in that image.

So, we decided to associate these images to an XML file that is related to another image which only has microcalcifications and not MASS regions which, in the end, will not produce any result.

## Exporting labels

#### Bounding boxes and problems with formats

A bounding box is a simple rectangle which delimits a region of interest. To define these bounding boxes there are two possible approaches:

* PASCAL VOC: [x_min, y_min, x_max, y_max]. x_min e y_min
are coordinates that point out the bottom left vertix of the rectangle, while  x_max e y_max point out the top right vertix.

* YOLO: [x_center, y_center, width, height]. x_center e y_center are the normalized coordinates of the center of the bounding box, followed by the respective heigth and width.

To convert YOLO to PASCAL VOC and viceversa, it is important to have the dimension of the image in order to perform normalization.

In this case we decided to use the YOLO method and so we saved bounding boxes in the YOLO format.


### Exporting labels with MASS

he rationale behind the process is the following: starting from *x, y, w, h* information about the bounding boxes containing ROIs will be obtained, which are not normalized as YOLO requires. So, normalization is applied and a list is created. The first value of this list is always zero, which identifies the class to which the image belongs to and, as we are working only on mass, the value will always be zero.

This list is then written as txt file and saved with the dataset containing info about images. To learn how to organize folders to train YOLO see the following [link](https://medium.com/@mgupta70/computer-vision-blogs-object-detection-with-yolov8-train-your-own-custom-object-detection-model-4a0fd3aaee59).

```
import matplotlib.patches as patches

patid = 53582683   #image code
breast, synthetized, mass_mask = synthetized_images(patid)
#cv2.imwrite(folderpath + '/' + str(patid) + '.png', synthetized)

a=load_inbreast_mask(XML_PATH + str(patid) + '.xml')
b=mask_to_yolo(a)
bb=bbox_to_txt(b)
if len(bb)>0:

  if len(bb)>100:
    print('PROBABILE PRESENZA DI DUE MASS', patid)

  else:
    cnts, _ = cv2.findContours(mass_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    image_height, image_width, _ = synthetized.shape

    x, y, w, h = cv2.boundingRect(cnt)

    label_list = [0, ((x+(w/2))/image_width), ((y+(h/2))/image_height), (w/image_width), (h/image_height)]

    plt.imshow(synthetized, interpolation='nearest')
    #rect = patches.Rectangle((((label_list[0]-(w/2))*image_width), ((label_list[1]-(h/2))*image_width)), w, h, linewidth=10, edgecolor='r', facecolor='none')

    a = [((label_list[1]-(label_list[3]/2))*image_width), ((label_list[1]-(label_list[3]/2))*image_width) ] #xmin, xmin
    b = [((label_list[2]-(label_list[4]/2))*image_height), ((label_list[2]+(label_list[4]/2))*image_height)] #ymin, ymax
    c = [((label_list[1]-(label_list[3]/2))*image_width), ((label_list[1]+(label_list[3]/2))*image_width) ] #xmin, xmax
    d = [((label_list[2]-(label_list[4]/2))*image_height), ((label_list[2]-(label_list[4]/2))*image_height)] #ymin, ymin
    e = [((label_list[2]+(label_list[4]/2))*image_height), ((label_list[2]+(label_list[4]/2))*image_height)] #ymax, ymax
    f = [((label_list[1]+(label_list[3]/2))*image_width), ((label_list[1]+(label_list[3]/2))*image_width) ] #xmax, xmax

    string_label_list = ' '.join(str(lab) for lab in label_list)

    #open text file
    text_file = open(LABEL_PATH + str(patid) + '.txt', "w")
    #write string to file
    text_file.write(string_label_list)
    #close file
    text_file.close()

    #DISEGNO RETTANGOLO
    plt.plot(a,b, color="red", linewidth=2)  #xmin, ymax
    plt.plot(c, d, color="red", linewidth=2)  #xmin, ymax
    plt.plot(c, e, color="red", linewidth=2)  #xmin, ymax
    plt.plot(f, b, color="red", linewidth=2)  #xmin, ymax
    plt.show()

```

### Exporting images with two mass labels

patid = 51049107   #INSERIRE CODICE IMMAGINE
breast, synthetized, mass_mask = synthetized_images(patid)
#cv2.imwrite(folderpath + '/' + str(patid) + '.png', synthetized)

a=load_inbreast_mask(XML_PATH + str(patid) + '.xml')
b=mask_to_yolo(a)
bb=bbox_to_txt(b)

cnts, _ = cv2.findContours(mass_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt1 = min(cnts, key = cv2.contourArea)
cnt2 = max(cnts, key = cv2.contourArea)

image_height, image_width, _ = synthetized.shape

x1, y1, w1, h1 = cv2.boundingRect(cnt1)
x2, y2, w2, h2 = cv2.boundingRect(cnt2)

label_list1 = [0, ((x1+(w1/2))/image_width), ((y1+(h1/2))/image_height), (w1/image_width), (h1/image_height)]
label_list2 = [0, ((x2+(w2/2))/image_width), ((y2+(h2/2))/image_height), (w2/image_width), (h2/image_height)]


plt.imshow(synthetized, interpolation='nearest')
    #rect = patches.Rectangle((((label_list[0]-(w/2))*image_width), ((label_list[1]-(h/2))*image_width)), w, h, linewidth=10, edgecolor='r', facecolor='none')

a1 = [((label_list1[1]-(label_list1[3]/2))*image_width), ((label_list1[1]-(label_list1[3]/2))*image_width) ] #xmin, xmin
b1 = [((label_list1[2]-(label_list1[4]/2))*image_height), ((label_list1[2]+(label_list1[4]/2))*image_height)] #ymin, ymax
c1 = [((label_list1[1]-(label_list1[3]/2))*image_width), ((label_list1[1]+(label_list1[3]/2))*image_width) ] #xmin, xmax
d1 = [((label_list1[2]-(label_list1[4]/2))*image_height), ((label_list1[2]-(label_list1[4]/2))*image_height)] #ymin, ymin
e1 = [((label_list1[2]+(label_list1[4]/2))*image_height), ((label_list1[2]+(label_list1[4]/2))*image_height)] #ymax, ymax
f1 = [((label_list1[1]+(label_list1[3]/2))*image_width), ((label_list1[1]+(label_list1[3]/2))*image_width) ] #xmax, xmax

a2 = [((label_list2[1]-(label_list2[3]/2))*image_width), ((label_list2[1]-(label_list2[3]/2))*image_width) ] #xmin, xmin
b2 = [((label_list2[2]-(label_list2[4]/2))*image_height), ((label_list2[2]+(label_list2[4]/2))*image_height)] #ymin, ymax
c2 = [((label_list2[1]-(label_list2[3]/2))*image_width), ((label_list2[1]+(label_list2[3]/2))*image_width) ] #xmin, xmax
d2 = [((label_list2[2]-(label_list2[4]/2))*image_height), ((label_list2[2]-(label_list2[4]/2))*image_height)] #ymin, ymin
e2 = [((label_list2[2]+(label_list2[4]/2))*image_height), ((label_list2[2]+(label_list2[4]/2))*image_height)] #ymax, ymax
f2 = [((label_list2[1]+(label_list2[3]/2))*image_width), ((label_list2[1]+(label_list2[3]/2))*image_width) ] #xmax, xmax

string_label_list1 = ' '.join(str(lab) for lab in label_list1)
string_label_list2 = ' '.join(str(lab) for lab in label_list2)


    #open text file
text_file = open(LABEL_PATH + str(patid) + '.txt', "w")
#write string to file
text_file.write(string_label_list1)
text_file.write('\n')
text_file.write(string_label_list2)
    #close file
text_file.close()

    #DISEGNO RETTANGOLO
plt.plot(a1,b1, color="red", linewidth=2)  #xmin, ymax
plt.plot(c1, d1, color="red", linewidth=2)  #xmin, ymax
plt.plot(c1, e1, color="red", linewidth=2)  #xmin, ymax
plt.plot(f1, b1, color="red", linewidth=2)  #xmin, ymax

plt.plot(a2,b2, color="red", linewidth=2)  #xmin, ymax
plt.plot(c2, d2, color="red", linewidth=2)  #xmin, ymax
plt.plot(c2, e2, color="red", linewidth=2)  #xmin, ymax
plt.plot(f2, b2, color="red", linewidth=2)  #xmin, ymax

plt.show()

## Splitting data

* train: 326
* validation: 41
* test: 41

Number of images in each group:

* train: 85
* validation: 9
* test: 12

Note: two images had a positive diagnosis but did not have an associated label: 22614097, 22614150. For this reason, those images were removed from the dataset.

## Data Augmentation

[*Albumentations*](https://albumentations.ai/) library is used

Note: This library uses bounding boxes in the Pascal Voc format.

As explained previously, the number of images used for the training phase are 326, and 85 of which have a MASS. So, we decide to apply data augmentation to improve training performances, in particular, the following options were used:

* zoom = 1.3
* scale = 0.8 and rotate = 0.45
* scale = 0.8 and rotate = -0.45

Subsequently, we performed image augmentation of all images with flipping, in order to obtain a balanced dataset.

## YOLO Training

YOLOv5 is used and training is carried out in 200 epochs. For a matter of time, we decided to split the training into 4 steps of 50 epochs.

```
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

import torch #import pytorch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

!python train.py --img 2000 --batch 8 --epochs 50  --data breastcancer_yolo.yaml --weights last.pt  --workers 2

```


## YOLO Detect

Weights obtained from the last tranining step are used. After several tries, we decided to set the threshold for confidence score to 0.008. Moreover, it is possible to save ROIs' coordinates with the 'save-txt' option and results will appear in the 'results' folder.

```
!python yolov5/detect.py --source /content/drive/MyDrive/tesi/esperimenti/archive/INbreast/YOLODataset/images/test --weights yolov5/last.pt --conf 0.008 --iou 0.20 --augment --save-txt

```

## Exporting for classification

ROI's export is performed, which will be followed by data augmentation. The associated ROI is cropped and saved as '.png' image in specific folder, where the training dataset will be stored.

##### Images with only one ROI

```
patient_id = 53587663


label_path = train_path + 'labels/Train/' + str(patient_id) + '.txt'
image_path = train_path + 'images/Train/' + str(patient_id) + '.png'


file = open(label_path, "r")
num_roi = []
label = []
stringlist = []
label_list = []
for line in file.readlines():
    num_roi.append(1) #useful to see how many ROIs there are, if len>1 there are two ROIs
    stringlist = line.split(sep=None, maxsplit=-1)  #creates a lists of strings from the txt file
    for i in stringlist:           #transorms strings list into float format
     label_list.append(float(i))

    label_list.remove(0.0)         #removes first element of the list, which is the class the image belongs to
    print(label_list)
file.close()

np_image = np.array(Image.open(image_path).convert("RGB"))
image_height, image_width, _ = np_image.shape

x_min = int((label_list[0] - (label_list[2] / 2)) * image_width)
y_min = int((label_list[1] - (label_list[3] / 2)) * image_height)
x_max = int((label_list[0] + (label_list[2] / 2)) * image_width)
y_max = int((label_list[1] + (label_list[3] / 2)) * image_height)

roi = np_image[y_min: y_max, x_min:x_max ]

cv2.imwrite(export_path + '/' + str(patient_id) + '.png', roi)

plt.imshow(np_image, interpolation='nearest')

plt.plot(x_min, y_min, marker='.', color="red")
plt.plot(x_min, y_max, marker='.', color="red")
plt.plot(x_max, y_max, marker='.', color="red")
plt.plot(x_max, y_min, marker='.', color="red")
plt.show()

plt.imshow(roi, interpolation='nearest')
```

##### Images with two ROIs

```
patient_id = 51049107

label_path = train_path + 'labels/Train/' + str(patient_id) + '.txt'
image_path = train_path + 'images/Train/' + str(patient_id) + '.png'

file = open(label_path, "r")
num_roi = []
label = []
stringlist = []
label_list = []
for line in file.readlines():
    num_roi.append(1) #useful to see how many ROIs there are, if len>1 there are two ROIs
    stringlist = line.split(sep=None, maxsplit=-1) #creates a lists of strings from the txt file
    for i in stringlist:            #transorms strings list into float format
     label_list.append(float(i))

    label_list.remove(0.0)        #removes first element of the list, which is the class the image belongs to
  #  print(label_list)
file.close()

np_image = np.array(Image.open(image_path).convert("RGB"))
image_height, image_width, _ = np_image.shape

x_min1 = int((label_list[0] - (label_list[2] / 2)) * image_width)
y_min1 = int((label_list[1] - (label_list[3] / 2)) * image_height)
x_max1 = int((label_list[0] + (label_list[2] / 2)) * image_width)
y_max1 = int((label_list[1] + (label_list[3] / 2)) * image_height)
x_min2 = int((label_list[4] - (label_list[6] / 2)) * image_width)
y_min2 = int((label_list[5] - (label_list[7] / 2)) * image_height)
x_max2 = int((label_list[4] + (label_list[6] / 2)) * image_width)
y_max2 = int((label_list[5] + (label_list[7] / 2)) * image_height)

roi1 = np_image[y_min1: y_max1, x_min1:x_max1]
roi2 = np_image[y_min2: y_max2, x_min2:x_max2]

cv2.imwrite(export_path + '/' + str(patient_id) + '_mass1.png', roi1)
cv2.imwrite(export_path + '/' + str(patient_id) + '_mass2.png', roi2)


plt.imshow(np_image, interpolation='nearest')

plt.plot(x_min1, y_min1, marker='.', color="red")
plt.plot(x_min1, y_max1, marker='.', color="red")
plt.plot(x_max1, y_max1, marker='.', color="red")
plt.plot(x_max1, y_min1, marker='.', color="red")
plt.plot(x_min2, y_min2, marker='.', color="red")
plt.plot(x_min2, y_max2, marker='.', color="red")
plt.plot(x_max2, y_max2, marker='.', color="red")
plt.plot(x_max2, y_min2, marker='.', color="red")
plt.show()

plt.imshow(roi1, interpolation='nearest')
plt.show()

plt.imshow(roi2, interpolation='nearest')
plt.show()
```

### Data Augmentation on ROIs

In this case too, Data Augmentation is applied with zoom in, rotation and flip. Images are then saved and stored in the training folder.

```
patient_id = 20586934

image_path = import_nonaug + '/' + str(patient_id) + '.png'

image = cv2.imread(image_path)
height, width = image.shape[:2]
rotation_matrix1 = cv2.getRotationMatrix2D((width/2, height/2), 45, 0.85)
rotated_image1 = cv2.warpAffine(image, rotation_matrix1, (width, height))
rotation_matrix2 = cv2.getRotationMatrix2D((width/2, height/2), -45, 0.90)
rotated_image2 = cv2.warpAffine(image, rotation_matrix2, (width, height))

rotation_matrix3 = cv2.getRotationMatrix2D((width/2, height/2), 0, 1.3)
zoomed_image = cv2.warpAffine(image, rotation_matrix3, (width, height))

cv2.imwrite(export_path + '/' + str(patient_id) + '_1.png', rotated_image1)


flipped_image1 = cv2.flip(image, 3)
flipped_image2 = cv2.flip(image, -1)


f, axarr = plt.subplots(2,3)
axarr[0,0].imshow(image)
axarr[0,1].imshow(rotated_image1)
axarr[0,2].imshow(rotated_image2)
axarr[1,0].imshow(zoomed_image)
axarr[1,1].imshow(flipped_image1)
axarr[1,2].imshow(flipped_image2)
```

Analysis of ROIs' dimensions
max height: 984
max width: 976

### Images without ROIs

Rationale: choose height and width of areas without cancer, randomly, sampling dimensions from the minimum to the maximum of heights of all possible ROIs. This is done in such a way that dimensions of areas containing breast tissue will not be considered as a discriminant factor for classification.

```
lista_immagini = os.listdir(path_immagini)

for i in lista_immagini:
  size = len(i)
  patient_id = i[:size - 4]
  image_path = path_immagini + '/' + i
  np_image = np.array(Image.open(image_path).convert("RGB"))
  image_height, image_width, _ = np_image.shape

  HW_index = random.randint(0,91)

  roi_width = lista_widths[HW_index]
  roi_height = lista_heights[HW_index]

  start_row = random.randint(0, (image_height-roi_height))
  end_row = start_row + roi_height
  start_col = random.randint(0, (image_width-roi_width))
  end_col = start_col + roi_width

  area = np_image[start_row:end_row,start_col:end_col]

  #save area in the folder containing training images
  cv2.imwrite(export_path + '/' + str(patient_id) + '_1.png', area)

  print(patient_id)

  #plot selected image with associated area
  plt.imshow(np_image)

  plt.plot(start_col, start_row, marker='.', color="red")
  plt.plot(start_col, end_row, marker='.', color="red")
  plt.plot(end_col, start_row, marker='.', color="red")
  plt.plot(end_col, end_row, marker='.', color="red")

  plt.show()
```

#### Padding

In the end, padding is performed, in order to have images of the same dimensions. Black borders are added to images that are smaller than the biggest image in the dataset.


```
max_height = 984
max_width = 976

lista_images = os.listdir(import_path)

for i in lista_images:
  size = len(i)
  patient_id = i[:size - 4]
  image_path = import_path + '/' + i
  np_image = np.array(Image.open(image_path).convert("RGB"))
  roi_height, roi_width, _ = np_image.shape

  diff_height = max_height - roi_height
  diff_width = max_width - roi_width
  padded = cv2.copyMakeBorder(np_image, math.ceil(diff_height/2), math.floor(diff_height/2),
                               math.floor(diff_width/2),  math.ceil(diff_width/2),
                     cv2.BORDER_CONSTANT, None, value = 0)

  cv2.imwrite(export_path + '/' + str(patient_id) + '.png', cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
```
The same process is carried out on test and validation images, without data augmentation.

# Image Classification

Results of three different neural networks are compared: VGG19, ResNet50 e InceptionV3. Training of Inception is reported.

## Import of dependencies

```
import keras
import tensorflow as tf

from keras.models import Model
from keras.utils import plot_model
from keras.models import Sequential
from keras.applications import VGG19, VGG16
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

```

## Import and normalization of images

```
traindir = '/content/gdrive/MyDrive/tesi/esperimenti/Teresa/08_22_ClassPadded/train'
train_data = tf.keras.utils.image_dataset_from_directory(traindir,
                                               image_size=(984,976), # 1800,1200
                                               batch_size = 5,
                                               color_mode='rgb')
valdir = '/content/gdrive/MyDrive/tesi/esperimenti/Teresa/08_22_ClassPadded/val'
val_data = tf.keras.utils.image_dataset_from_directory(valdir,
                                               image_size=(984,976), # 1800,1200
                                               batch_size = 5,
                                               color_mode='rgb')
testdir = '/content/gdrive/MyDrive/tesi/esperimenti/Teresa/08_22_ClassPadded/test'
test_data = tf.keras.utils.image_dataset_from_directory(testdir,
                                               image_size=(984,976), # 1800,1200
                                               batch_size = 5,
                                               color_mode='rgb')

train_scaled = train_data.map(lambda x, y: (x/255,y))
val_scaled = val_data.map(lambda x, y: (x/255,y))
test_scaled = test_data.map(lambda x, y: (x/255,y))

```

## Loading and training of the model

```
image_size = [984,976]
#

inception = InceptionV3(input_shape = (image_size + [3]),
                        weights = 'imagenet',
                        include_top = False)

#
for layer in inception.layers:
  layer.trainable = False
#
x = Flatten()(inception.output)                          #flatten the output given by the previous layers
y = Dense(256, activation = 'relu')(x)   #add layer to lower number of parameters difference between layers
z = Dense(50, activation = 'relu') (y)
prediction = Dense(1, activation = "sigmoid")(z)     #add the two categories in the last layer, and use the softmax activation function
#
model = Model(inputs = inception.input, outputs = prediction)
model.summary()

#compiling the model
model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

logdir = '/content/gdrive/MyDrive/tesi/esperimenti/Teresa/08_22_TrainingLogs'
tensorboard_callback = keras.callbacks.TensorBoard(log_dir = logdir)

hist = model.fit(train_scaled,
          epochs = 150,
        #  batch_size = 20,
          validation_data = val_scaled,
          callbacks = [tensorboard_callback])

```

## Testing classifier on ROIs

```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn

predicted=model.predict(test_scaled)
y = np.concatenate([y for x, y in test_scaled], axis=0)
mat=(confusion_matrix(y,predicted.round()))
print(mat)

```


# Comments about results

### YOLO

Results obtained from YOLO on the 41 training images are:

* 16 True negatives
* 13 False positives
* 2 False negatives
* 7 True positives, where one of the regions of interest found actually contained cancer lesion.
* 3 True positives where regions of interest were in a different area than the actual one.

### Inception or ROIs

Results obtained from the classifier Inception on the 66 images identified from YOLO:

![image](https://github.com/TeresaDadda/BC_Malignancy_Investigation/assets/150792320/4b48ec86-4863-40ce-a79e-0ff85ebc9fd2)

### Pipeline of OD + IC

Results obtained from the entire pipeline on the 41 test images:

* 16 images were not detected and were actually negative.
* 3 positive images were not detected.
* 7 negative images where OD signals regions of interest (FP) correctly classified as *'not lesions'* from the classifier.
* 3 positive images where OD identifies wrong ROIs and are classified as *'not lesions'* from the classifier.
* 1 positive image where different ROIs are identified, included the right one from OD, but the wrong one is labelled as *'lesion'*.
* 6 negative images where the OD detects presence of regions of interest, where at least one is correctly identified as *'lesion'* by IC.
* 5 positive images where the OD detects the presence of different ROIs, where at least one is correct. The region containing the lesion is correctly labelled as *'lesion'* while the others are classified as *'not lesion'*.


