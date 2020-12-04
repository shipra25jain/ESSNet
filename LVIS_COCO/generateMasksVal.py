from pycocotools.coco import COCO
import lvis
import pickle
import numpy as np
import imageio as io
import numpy
lvispath = 'lvisdataset/lvis_v1_val.json'
cocopath = 'cocodataset/annotations/stuff_train2017.json'
cocopath_val = 'cocodataset/annotations/stuff_val2017.json'
#cocopath = '/scratch/knurrhahn/shijain/cocodataset/annotations/stuff_tesr2017.json'
lv = lvis.LVIS(lvispath)
coco = COCO(cocopath)
coco_val = COCO(cocopath_val)
coco2lvispath =  'coco2lvis.pkl'
coco2lvis = pickle.load(open(coco2lvispath,'rb'))
lvis_images = list(lv.imgs.keys())
coco_images = list(coco.imgs.keys())
coco_val_images = list(coco_val.imgs.keys())
notInCoco =  []
import tqdm
import pickle

ct = 0
noAnnotations = []
import imageio
for img in lvis_images:
	if(img not in coco_images and img not in coco_val_images):
		notInCoco.append(img)
		continue
	ann_ids = lv.get_ann_ids(img_ids=[img])

	i = 0
	hasLvis = False
	for a in ann_ids:
		hasLvis = True

		if(i==0):
			mask = lv.anns[a]['category_id']*lv.ann_to_mask(lv.anns[a])
			addMask = (mask==0)

		else : 

			mask = mask + addMask*lv.anns[a]['category_id']*lv.ann_to_mask(lv.anns[a])
			addMask = (mask==0)
		i = i + 1
	inTrain = True
	if(img not in coco_images):
		inTrain = False
	if inTrain:
		coco_ann_ids  =   coco.getAnnIds(imgIds=[img])
	else:
		coco_ann_ids = coco_val.getAnnIds(imgIds=[img])
	i = 0
	if(len(coco_ann_ids)==0):
		print("empty")
	#print("coco")
	hasCoco = False
	for c in coco_ann_ids:
		hasCoco = True

		if(i==0):
			if inTrain:
				cmask = coco.anns[c]['category_id']*coco.annToMask(coco.anns[c])
				caddMask = (cmask==0)
			else:
				cmask = coco_val.anns[c]['category_id']*coco_val.annToMask(coco_val.anns[c])
				caddMask = (cmask==0)
		else:
			if inTrain:
				cmask = cmask + caddMask*coco.anns[c]['category_id']*coco.annToMask(coco.anns[c])
				caddMask = (cmask==0)
			else:
				cmask = cmask + caddMask*coco_val.anns[c]['category_id']*coco_val.annToMask(coco_val.anns[c])
				caddMask = (cmask==0)
		i= i+1
	for k, v in coco2lvis.items():
		cmask[cmask==k] = v
	if(hasCoco and hasLvis):
		finalMask = mask + addMask*cmask
	elif(hasCoco):
		finalMask = cmask
	elif(hasLvis):
		finalMask = mask
	else:
		noAnnotations.append(img)
		continue
	ct  = ct + 1
	maskArray =  finalMask.astype(numpy.uint16)
	io.imwrite('lvisdataset/lvis_mask/'+str(img)+'.png', maskArray)

print("images not in coco")
print(notInCoco)
print("images with no annotations")
print(noAnnotations)
