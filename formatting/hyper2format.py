# ref: 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

import json
import numpy as np
import os
from pathlib import PurePosixPath as GPath
import pathlib
from typing import Text, Union
PathType = Union[Text, pathlib.PurePosixPath]
from PIL import Image
import torch
import math
import shutil
from argparse import ArgumentParser
import sys


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def from_json(path: PathType):
    """Loads a JSON camera into memory."""
    path = GPath(path)
    # with path.open('r') as fp:
    with open(path, 'r') as fp:
      camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
      camera_json['tangential_distortion'] = camera_json['tangential']

    camera = {}
    camera['orientation']=np.asarray(camera_json['orientation'])
    camera['position']=np.asarray(camera_json['position'])
    camera['focal_length']=camera_json['focal_length']
    camera['principal_point']=np.asarray(camera_json['principal_point'])
    camera['skew']=camera_json['skew']
    camera['pixel_aspect_ratio']=camera_json['pixel_aspect_ratio']
    camera['radial_distortion']=np.asarray(camera_json['radial_distortion'])
    camera['tangential_distortion']=np.asarray(camera_json['tangential_distortion'])
    camera['image_size']=np.asarray(camera_json['image_size'])
    return camera


parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--datadir", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args(sys.argv[1:])
output_path = args.output_path
datadir = args.datadir
ratio = 0.5     # size of input image


with open(f'{datadir}/scene.json', 'r') as f:
    scene_json = json.load(f)
with open(f'{datadir}/metadata.json', 'r') as f:
    meta_json = json.load(f)
with open(f'{datadir}/dataset.json', 'r') as f:
    dataset_json = json.load(f)
    
near = scene_json['near']
far = scene_json['far']
coord_scale = scene_json['scale']
scene_center = scene_json['center']
all_img = dataset_json['ids']
val_id = dataset_json['val_ids']

####################################### Split Dataset #####################################################
# train : test : val = 8 : 1 : 1
i_test = np.array([i for i in np.arange(len(all_img)) if (i%10 == 0)])
i_val = i_test + 5
i_val = i_val[:-1,]
i_train = np.array([i for i in np.arange(len(all_img)) if ((i not in i_test) and (i not in i_val))])
###########################################################################################################

all_cam = [meta_json[i]['camera_id'] for i in all_img]
all_time = [meta_json[i]['warp_id'] for i in all_img]
max_time = max(all_time)
all_time = [meta_json[i]['warp_id']/max_time for i in all_img]
selected_time = set(all_time)
max_time = max(all_time)
min_time = min(all_time)
i_video = [i for i in range(len(all_img))]
i_video.sort()
all_cam_params = []
for im in all_img:
    camera = from_json(f'{datadir}/camera/{im}.json')
    all_cam_params.append(camera)
idx_h = all_cam_params[0]['image_size'][1]
idx_w = all_cam_params[0]['image_size'][0]

all_depth = [f'{datadir}/depth/{int(1/ratio)}x/{i}.npy' for i in all_img]
all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in all_img]

train_transform = {}
test_transform = {}

def get_transforms_json(output_path, split, data, all_cam_params, all_img, idx_h, idx_w):
    if not os.path.exists(output_path+"/"+split):
        os.makedirs(output_path+"/"+split)

    transform = {}
    FovX = focal2fov(all_cam_params[0]['focal_length'], idx_w)
    FovY = focal2fov(all_cam_params[0]['focal_length'], idx_h)
    transform['camera_angle_x'] = FovX
    transform['camera_angle_y'] = FovY
    transform['frames'] = []

    for idx in data:
        frame = {}
        matrix = [
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 1]
		]
        camera = all_cam_params[idx]
        image = Image.open(all_img[idx])
        w = image.size[0]
        h = image.size[1]
        time = all_time[idx]
        R = camera['orientation'].T
        T = - camera['position'] @ R
        R[:,0] = -R[:,0]
        R = -np.transpose(R)
        R = np.vstack([R, np.array([0,0,0])])
        T = -T
        T = np.hstack([T, np.array([1])])
        matrix = np.linalg.inv(np.column_stack([R, T]))
        # opengl -> opencl
        matrix[1:3,:] = - matrix[1:3,:]
        image_path = "./"+split+"/frame"+all_img[idx].split("/")[-1]

        frame['file_path'] = image_path
        frame['time'] = time
        frame['height'] = h
        frame['width'] = w
        frame['transform_matrix'] = matrix.tolist()
        transform['frames'].append(frame)
        shutil.copy(all_img[idx], output_path+"/"+split+"/frame"+all_img[idx].split("/")[-1])

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path+"/"+split+"_transforms"+".json", "w") as outfile:
	    json.dump(transform, outfile, indent=4)

get_transforms_json(output_path, "train", i_train, all_cam_params, all_img, idx_h, idx_w)
get_transforms_json(output_path, "test", i_test, all_cam_params, all_img, idx_h, idx_w)
get_transforms_json(output_path, "val", i_val, all_cam_params, all_img, idx_h, idx_w)

print("done!")