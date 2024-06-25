import os 
import json 
import base64
from PIL import Image
import PIL 
import io
import cv2 
import numpy as np 
import math 
import random 
import shutil 
import sys 
import argparse

class labelme_to_yolo: 

    labelme_dir = ''
    delete_json = 'n'
    val_split = 0.1 
    test_split = 0.1 
    labels = []

    def __init__(self, labelme_dir, delete_json='n', val_split=0.10, test_split=0.10, labels=[]): 
        self.labelme_dir =  labelme_dir
        self.delete_json = delete_json 
        self.val_split = val_split 
        self.test_split = test_split 
        self.labels = labels 

    def run(self): 
        self.make_directories() # Generate train/test/valid directories with images/labels subdirectories 
        self.split_data() # Split data into valid/test/training data. Optional parameters to specify percentages for val/test. Default 10% each, 80% training. 
        # Convert labelme json to YOLOv5 PyTorch format 
        self.convert_annotations(self.labelme_dir + "/valid")
        self.convert_annotations(self.labelme_dir + "/test")
        self.convert_annotations(self.labelme_dir + "/train")
        # Generate YAML file for data 
        self.generate_yaml(self.labels, self.labelme_dir)

    # decode script from LabelMe. Decodes to (h, w, num channels) tuple 
    def img_b64_to_arr(self, img_b64):
        f = io.BytesIO()
        f.write(base64.b64decode(img_b64))
        img_arr = np.array(PIL.Image.open(f))
        return img_arr

    # Finds midpoint of box 
    def midpoint(self, x1, x2, y1, y2): 
        return ((x1 + x2)/2, (y1 + y2)/2)

    # normalizes pixel values 
    def normalize(self, val_tuple, w, h): 
        return (val_tuple[0]/w, val_tuple[1]/h)  

    # Gets dimensions of box 
    def dimensions(self, x1, x2, y1, y2): 
        return (abs(x2-x1), abs(y2-y1))

    # Converts labelme.json file to yolov5 annotation format .txt 
    def convert_annotations(self, folder_path):
        label_path = folder_path + "/labels"
        for filename in os.listdir(label_path): 
            file_path = os.path.join(label_path, filename)

            if filename.endswith('.json'): 
                name = filename.split('.json')[0]
                with open(file_path, 'r') as file: 
                    try: 
                        loadjson = json.load(file)

                        imageData = self.img_b64_to_arr(loadjson['imageData']) # load in image data 
                        height, width = imageData.shape[0], imageData.shape[1] # Get size of image for normalization
                        
                        for entry in loadjson['shapes']: # Loop through all points in annotation 
                            label = entry['label'] 

                            if(label not in self.labels): # Keep track of the labels/classes
                                self.labels.append(label)

                            pointarray = entry['points'] 

                            box = [item for sublist in pointarray for item in sublist] # Make a single array of bounding box coordinate points (x1, x2, y1, y2)
                            center = ' '.join(map(str, self.normalize(self.midpoint(box[0], box[2], box[1], box[3]), width, height))) # make string with coordinates of midpoint "x y"
                            dim = ' '.join(map(str, self.normalize(self.dimensions(box[0], box[2], box[1], box[3]), width, height))) # Make a string with dimensions "w h"

                            file_path = label_path + '/' + name + ".txt" 

                            with open(file_path, 'a') as file: # Write to txt file 
                                file.write(label + " " + center + " " + dim + "\n")

                        # Save image data to corresponding images directory 
                        img = Image.fromarray(imageData)
                        img.save(folder_path + "/images/" + name + ".jpg")

                        # Delete original json file 
                        json_path = label_path + "/" + filename
                        if(self.delete_json!='n'): 
                            os.remove(json_path)
                        else:
                            self.move_data([filename], label_path, self.labelme_dir + "/labelme_json")

                    except Exception as e: 
                        print(e)

    # Make test/train/valid directories 
    def make_directories(self): 
        os.makedirs(self.labelme_dir + "/test", exist_ok=True)
        os.makedirs(self.labelme_dir + "/train", exist_ok=True)
        os.makedirs(self.labelme_dir + "/valid", exist_ok=True)

        os.makedirs(self.labelme_dir + "/test/images", exist_ok=True)
        os.makedirs(self.labelme_dir + "/test/labels", exist_ok=True)

        os.makedirs(self.labelme_dir + "/train/images", exist_ok=True)
        os.makedirs(self.labelme_dir + "/train/labels", exist_ok=True)

        os.makedirs(self.labelme_dir + "/valid/images", exist_ok=True)
        os.makedirs(self.labelme_dir + "/valid/labels", exist_ok=True)

        if(self.delete_json=='n'): 
            os.makedirs(self.labelme_dir + "/labelme_json", exist_ok=True)
            

    # Move file list from src to target directory 
    def move_data(self, file_array, input_folder, dest_folder): 
        for f in file_array: 
            src = os.path.join(input_folder, f)
            dest = os.path.join(dest_folder, f)
            shutil.move(src, dest)

    # Split file array into sections. Ratio depends on percentage of validation/testing datasets 
    def split_data(self): 
        json_files = [f for f in os.listdir(self.labelme_dir) if f.endswith(".json")]  # Get all .json annotations
        random.shuffle(json_files) # Shuffle list randomly 

        total_files = len(json_files) # total # of json annotations 
        split1 = int(total_files*self.val_split)
        split2 = split1 + int(total_files*self.test_split)

        # 3 arrays: validation files, testing files, training files 
        val_split = json_files[:split1]
        test_split = json_files[split1:split2]
        train_split = json_files[split2:]

        # Move data into target directories 

        validpath = self.labelme_dir + "/valid/labels"
        testpath = self.labelme_dir + "/test/labels"
        trainpath = self.labelme_dir + "/train/labels"

        self.move_data(val_split, self.labelme_dir, validpath)
        self.move_data(test_split, self.labelme_dir, testpath)
        self.move_data(train_split, self.labelme_dir, trainpath)

    # Generate yaml file based on saved labels for classes 
    def generate_yaml(self, label_arr, folder_path): 
        f = open(folder_path + "/data.yaml", "a")
        f.write("names:\n")
        for label in label_arr: 
            f.write("- " + "'" + label + "'" + "\n")
        f.write("nc: " + str(len(label_arr)) + "\n")
        f.write("test: " + folder_path + "/test/images" + "\n") 
        f.write("train: " + folder_path + "/train/images" + "\n") 
        f.write("val: " + folder_path + "/val/images") 



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme_dir', type=str, help='Specify the path of the labelme json files.')

    parser.add_argument('--delete_json', type=str, default='n', help='input the letter y if you would like to delete the labelmejson files after conversion.')

    parser.add_argument('--val_split', type=float, default=0.10, help='Specify the validation input size as a number between 0 and 1. Default 0.1 (10%)')

    parser.add_argument('--test_split', type=float, default=0.10, help='Specify the test input size as a number between 0 and 1. Default 0.1 (10%)')

    args = parser.parse_args()

    if((not (args.test_split >= 0 and args.test_split <= 1)) or (not (args.val_split >= 0 and args.val_split <= 1)) or (not ((args.test_split + args.val_split) == 1))):
        print("Val and test split must be in between 0 and 1 and sum to 1.")
    elif(not os.path.exists(args.labelme_dir)): 
        print("File path does not exist.")
    else: 
        do_conversion = labelme_to_yolo(labelme_dir=args.labelme_dir, delete_json=args.delete_json, val_split=args.val_split, test_split=args.test_split, labels=[])
        do_conversion.run()
