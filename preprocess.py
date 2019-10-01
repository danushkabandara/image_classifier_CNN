#read the images and preform data file and crop the images to get POI(Preform of Interest) images for CNN
#labels: 0 loss good, above 2.609 bad(balance the labels into zeros and above threshold values)


#preform_height_mm= bottom segments bottom edge value-top segments top edge value
#image size can vary so read image height in pixels and do the linear mapping from mm to pixel
#save as 1000 pixel height


import cv2
import csv
import pandas
import yaml
import numpy as np

import os

#hash
IMAGE_WIDTH = 4161
PFA_DIR = r""
IMAGES_DIR = r""
PREFORM_SETTINGS_HASH = "7ad562d2b3f740496ca3c0c59d346adaf996596d"
IMAGE_SETTINGS_HASH = "eee6eef6130ec3284e043072e0caeb30a0ccbf14"

def main():
    labels_file =  open(PFA_DIR+"output\\labels.csv",'a')
    #read csv containing PFA id's matched with PFA labels and preform of interest
    df = pandas.read_csv(PFA_DIR + PREFORM_SETTINGS_HASH + "\\preform_data.csv", index_col ='serial_handle_id')
    for index, row in df.iterrows():
       try:
           img = cv2.imread(IMAGES_DIR + IMAGE_SETTINGS_HASH + "\\" + index + "\\preform.png")
       except:
           print ("Failed to read image with ID : "+ index)
           continue
       #get the actual height of part of the preform shown in the image in mm 
       top_segment, bottom_segment = load_bottom_top_segments(IMAGES_DIR + IMAGE_SETTINGS_HASH + "\\" + index + "\\stack.yaml")
       if top_segment == None or top_bottom_segment < 5: #continue execution if file is missing or stacked images are missing more than 3/4 ths of the preform
           continue
       preform_height_mm, top_edge_of_top_seg = mm_height_of_preform_image(PFA_DIR + PREFORM_SETTINGS_HASH + "\\preform_data.yaml", bottom_segment, top_segment)
       if preform_height_mm == None:
           continue
       #get the image dimensions in pixels and create a linear mapping from mm to pixels
       image_height, image_width, image_channels = img.shape
       end_poi = int(convert_poi_val_to_pixel(row['start_foi'], preform_height_mm, image_height, top_edge_of_top_seg ))
       start_poi = int (convert_poi_val_to_pixel(row['end_foi'], preform_height_mm, image_height, top_edge_of_top_seg))

       try:
           crop_img = img[start_poi:end_poi, 0:IMAGE_WIDTH]
       except:
           print ("Cannot create image crop for id: "+index)
           break
       
       cv2.imwrite(PFA_DIR+"output\\"+index+".png", crop_img)
       labels_file.write(index+","+ str(row['foi_loss_length_km'])+ '\n')
       labels_file.flush() #in case of exception, always write file buffer before continuing loop
    

#loads yaml file
def yaml_loader(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as yaml_file:
            data = yaml.load(yaml_file)
            return data
    else:
        print("file doesnt exist for filepath: " + filepath)
        return None


#loads segment data from preform.yaml to calculate the mm distance showed by the stacked preform image
def mm_height_of_preform_image(filepath, bottom, top):
    data = yaml_loader(filepath)
    if data == None:
        return None, None
    index_of_bottom  = data.get('PFA_segments').get('name').index(bottom)
    index_of_top = data.get('PFA_segments').get('name').index(top)
    bottom_edge_of_bottom_seg = data.get('PFA_segments').get('bottom')[index_of_bottom]
    top_edge_of_top_seg = data.get('PFA_segments').get('top')[index_of_top]
    return (bottom_edge_of_bottom_seg-top_edge_of_top_seg), top_edge_of_top_seg



#load the bottom and top segments using the stack.yaml file
def load_bottom_top_segments(filepath):
    data = yaml_loader(filepath)
    if data == None:
        return None, None
    top_segment = data.get('top_segment')
    bottom_segment = data.get('bottom_segment')
    return top_segment, bottom_segment


def convert_poi_val_to_pixel(poi_val, preform_height_mm, image_height, top_edge_of_top_seg):
    pixel_val = (image_height/preform_height_mm)*(poi_val-top_edge_of_top_seg)
    return pixel_val


def test():
    top_segment, bottom_segment = load_bottom_top_segments(PFA_DIR+IMAGE_SETTINGS_HASH+"\\AA-08-NK-71"+"\\stack.yaml")
    print(top_segment)
    print(bottom_segment)
    mm_height_of_preform_image(PFA_DIR+PREFORM_SETTINGS_HASH+"\\preform_data.yaml", bottom_segment, top_segment)

if __name__ == "__main__":
    main()
