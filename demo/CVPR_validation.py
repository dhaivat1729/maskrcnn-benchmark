import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import time
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import glob

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
	cfg,
	min_image_size=800,
	confidence_threshold=0.7,
)

def load(image_path):
	"""
	Given an url of an image, downloads the image and
	returns a PIL image
	"""
	pil_image = Image.open(image_path).convert("RGB")
	# convert to BGR format
	image = np.array(pil_image)[:, :, [2, 1, 0]]
	return image

input_dir = '/network/tmp1/bhattdha/validation_data/frames/000003/'
input_image_list = glob.glob(input_dir + '*.png')
input_image_list.sort()
i = 0

forward_pass_time = []

for image_path in input_image_list:
	image = load(image_path)
	output_dir = '/network/tmp1/bhattdha/validation_data/inference/000003/'
	curr_time = time.time()
	predictions = coco_demo.run_on_opencv_image(image)
	inference_time = time.time() - curr_time
	forward_pass_time.append(inference_time)
	im = Image.fromarray(predictions)
	im.save(output_dir + str(i).zfill(6) + '.png')
	i = i + 1


## Let's compute the average time
av_time = np.mean(forward_pass_time)
print("Average forward pass time is: ", av_time)

## Let's save the stats
data = {'input_dir':input_dir,
		'output_dir':output_dir,
		'config_file':config_file,
		'forward_pass_time': forward_pass_time,
		'av_time':av_time
		}

np.save(output_dir + 'stats.npy', data)