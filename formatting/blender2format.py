import os
import json
from PIL import Image
import shutil
import numpy as np

data_path = "./data/dnerf/bouncingballs/"

def formatting(split):
    with open(data_path+"transforms_"+split+".json") as f:
        old_transforms = json.load(f)
    new_transforms = dict()
    new_transforms["camera_angle_x"] = old_transforms["camera_angle_x"]
    new_transforms["camera_angle_y"] = old_transforms["camera_angle_x"]
    new_transforms["frames"] = []

    for idx, frame in enumerate(old_transforms['frames']):
        new_frame = {}
        old_imgname = data_path+frame['file_path']+".png"
        new_imgname = "./"+split+"/frame"+str(idx).zfill(6)+".png"
        image = Image.open(old_imgname)
        shutil.copy(old_imgname, data_path+new_imgname)
        new_frame["file_path"] = new_imgname
        new_frame["time"] = frame["time"]
        new_frame["height"] = image.size[1]
        new_frame["width"] = image.size[0]
        matrix = frame["transform_matrix"]
        # opengl -> opencl
        matrix = np.array(matrix)
        matrix[1:3,:] = -matrix[1:3,:]
        new_frame["transform_matrix"] = matrix.tolist()

        os.remove(old_imgname)
        new_transforms["frames"].append(new_frame)
    
    with open(data_path+"/"+split+"_transforms"+".json", "w") as outfile:
	    json.dump(new_transforms, outfile, indent=4)
         
    os.remove(data_path+"transforms_"+split+".json")

formatting("train")
formatting("test")
formatting("val")

