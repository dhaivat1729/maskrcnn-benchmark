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
import sys
import os

## TO use the submission builder!
sys.path.append('/network/home/bhattdha/rvchallenge-starter-kit/')
import submission_builder

model_list = os.listdir('/network/home/bhattdha/maskrcnn-benchmark/configs/CVPR_model_tests/')


for file_name in model_list:

	# config_file = "../configs/caffe2/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml"
	config_file = "../configs/CVPR_model_tests/" + file_name

	# update the config options with the config file
	cfg.merge_from_file(config_file)
	# manual override some options
	cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

	coco_demo = COCODemo(
		cfg,
		min_image_size=400,
		confidence_threshold=0.5,
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

	def do_detection(image_path):
		image = load(image_path)
		# output_dir = '/network/tmp1/bhattdha/validation_data/inference/000003/'
		# curr_time = time.time()
		predictions = coco_demo.compute_prediction(image)
		top_predictions = coco_demo.select_top_predictions(predictions)
		
		return top_predictions

	test_dir = '/network/tmp1/bhattdha/test_data/test_data/'

	classes = [
	    'none',
	    'bottle',
	    'cup',
	    'knife',
	    'bowl',
	    'wine glass',
	    'fork',
	    'spoon',
	    'banana',
	    'apple',
	    'orange',
	    'cake',
	    'potted plant',
	    'mouse',
	    'keyboard',
	    'laptop',
	    'cell phone',
	    'book',
	    'clock',
	    'chair',
	    'dining table',
	    'couch',
	    'bed',
	    'toilet',
	    'television',
	    'microwave',
	    'toaster',
	    'refrigerator',
	    'oven',
	    'sink',
	    'person'
	]
	classes_np = np.array(classes)
	classes_arg = classes_np.reshape((len(classes), 1)).tolist()

	division_fact = [5,10,15,20]

	for i in range(len(division_fact)):
		submission_folder_name = 'test_submission_' + file_name[:-5] + '_' + str(division_fact[i])
		writer = submission_builder.SubmissionWriter(submission_folder_name, classes)
		for sequence_name in os.listdir(test_dir):
			if not sequence_name.endswith('.zip'):
				image_list = os.listdir(os.path.join(test_dir, sequence_name))
				image_list.sort()
				for image_file in image_list:
					detections = do_detection(os.path.join(test_dir, sequence_name, image_file))
					print("Detection is happening", os.path.join(test_dir, sequence_name, image_file))
					for index in range(detections.bbox.size()[0]):
						# let's get things necessary one at a time
						xmin = int(detections.bbox[index][0].item())
						ymin = int(detections.bbox[index][1].item())
						xmax = int(detections.bbox[index][2].item())
						ymax = int(detections.bbox[index][3].item())

						## Let's get the class probabilities, slighly tricky!

						## Initialize
						class_probabilities = np.zeros((len(classes),))
						class_probabilities = class_probabilities.tolist()
						detection_label = coco_demo.CATEGORIES[detections.get_field('labels')[index].item()]

						## If detection is part of our classes
						if detection_label in classes:
							leftover_prob = 1 - detections.get_field('scores')[index].item()
							class_probabilities = np.array(class_probabilities) + leftover_prob/(len(classes)-1)
							class_probabilities[classes.index(detection_label)] = detections.get_field('scores')[index].item()
						else:
							# class_probabilities = np.array(class_probabilities) + 1.0/len(classes)
							continue
							# class_probabilities[0] = 1.0

						x_cov = (xmax - xmin)/division_fact[i]
						y_cov = (ymax - ymin)/division_fact[i]
						# xy_cov = (x_cov*y_cov)**0.5
						xy_cov = 0
						upper_left_cov = [[x_cov,xy_cov],[xy_cov,x_cov]]
						lower_right_cov = [[y_cov,xy_cov],[xy_cov,y_cov]]
						# print(class_probabilities)
						writer.add_detection(class_probabilities, xmin, ymin, xmax, ymax, upper_left_cov=upper_left_cov, lower_right_cov=lower_right_cov)
					writer.next_image()
				writer.save_sequence(sequence_name)


# forward_pass_time = []




# ## Let's compute the average time
# av_time = np.mean(forward_pass_time)
# print("Average forward pass time is: ", av_time)

# ## Let's save the stats
# data = {'input_dir':input_dir,
#       'output_dir':output_dir,
#       'config_file':config_file,
#       'forward_pass_time': forward_pass_time,
#       'av_time':av_time
#       }

# np.save(output_dir + 'stats.npy', data)