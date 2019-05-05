'''
This file is used to generate small dataset for specific categories from large coco dataset.

This utility could be used to extract small dataset for our specific categories out of a larger dataset
'''

import json

## API functionalities to be used!
from pycocotools.coco import COCO 
from shutil import copyfile


## dataset directory
dataset_directory = '/network/tmp1/bhattdha/coco_dataset/train2017/'

## This is where new images only for our categories of interest will be stored
new_dataset_directory = '/network/tmp1/bhattdha/coco_dataset_new/train2017_modified/'

## annotations file path(contains json file)
## This has only been tested for instances_*.json type files. 
annotations_path = '/network/tmp1/bhattdha/coco_dataset/annotations/annotations/instances_train2017.json'

## this is where new file for annotations will be stored
new_annotations_path = '/network/tmp1/bhattdha/coco_dataset_new/annotations_modified/'

## Our classes
sub_classes = [
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
	'tv',
	'microwave',
	'toaster',
	'refrigerator',
	'oven',
	'sink',
	'person'
]

## TO use API, coco is constructor for COCO class
coco = COCO('/network/tmp1/bhattdha/coco_dataset/annotations/annotations/instances_train2017.json') 
alldata = coco.dataset

'''
alldata has 5 keys, dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
'info' and 'licenses' won't change at all! 

'''

## newdata same object as alldata, to be modified in future
newdata = alldata

### Let's get image ids for our categories. 
imgId_list = []
for cat in sub_classes:
	catIds = coco.getCatIds(catNms=cat)
	imgIds = coco.getImgIds(catIds=catIds)
	imgId_list = imgId_list + imgIds

## imgId_list may have multiple copies of same Ids, to get rid of that, we will find all unique values
def unique(in_list): 
	# intilize a null list 
	unique_list = [] 
	
	# traverse for all elements 
	for x in in_list: 
		# check if exists in unique_list or not 
		if x not in unique_list: 
			unique_list.append(x) 

	return unique_list

imgId_list = unique(imgId_list)


## We got all the unique IDs for our image, it's time to filter irrelavent images from alldata['images']
new_images_list = []
for i in range(len(alldata['images'])):
	if alldata['images'][i]['id'] in imgId_list:
		new_images_list.append(alldata['images'][i])

## done with newimages in data!!
newdata['images'] = new_images_list

### Let's get the annotations done!!
new_annotations = []
catIds = coco.getCatIds(catNms=sub_classes)
for i in range(len(alldata['annotations'])):
	## We will have to edit category_id in future(keep your hopes high!)
	if alldata['annotations'][i]['category_id'] in catIds:
		new_annotations.append(alldata['annotations'][i])

## Got annotations only for our thing
newdata['annotations'] = new_annotations

### Let's get done with alldata['categories']
new_categories = []
for i in range(len(alldata['categories'])):
	## Time to only keep our relevant categories!
	if alldata['categories'][i]['id'] in catIds:
		new_categories.append(alldata['categories'][i])

newdata['categories'] = new_categories

## let's map the new categories between 1 to num_categories. 
## This dictionary has mapping between old category ID and new category IDs
'''
new_ids[old_category_id] = new_category_id
'''

new_ids = {}
curr_id = 1
for i in range(len(newdata['categories'])):
	## Creating a dictionary mapping between new ID and old ID
	new_ids[newdata['categories'][i]['id']] = curr_id
	newdata['categories'][i]['id'] = curr_id
	curr_id = curr_id + 1

## Final editing in the annotation ids
for i in range(len(newdata['annotations'])):
	old_id = newdata['annotations'][i]['category_id']

	## The new ID will be mapped according to the old one from the dictionary
	newdata['annotations'][i]['category_id'] = new_ids[old_id]

##### Now we are done with creating all the content for our json file! So let's dump the goddamn thing

with open(new_annotations_path + 'instances_train2017_modified.json', 'w') as outfile:  
    json.dump(newdata, outfile)

print("json data stored successfully!")

### Now let's save all the necessary images which will form our dataset
for image in imgId_list:
	img_name = str(image).zfill(12) + '.jpg'
	src_image = dataset_directory + img_name
	dst_image = new_dataset_directory + img_name
	copyfile(src_image, dst_image)



