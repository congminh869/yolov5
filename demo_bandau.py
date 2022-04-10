import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device
from utils.datasets import letterbox
import os
import yaml
import torch.nn as nn
import scipy.io
from torchvision import datasets, models, transforms

def detect_obj(model, stride, names, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
  high, weight = img_detect.shape[:2] #(chieu cao, chieu rong, chieu sau)
  # print('********************')
  # print(high, weight) #444 640
  # print('********************')
  #####################################
  classify = False
  agnostic_nms = False
  augment = False
  # Set Dataloader
  #vid_path, vid_writer = None, None
  # Get names and colors

  count = 0
  t = time.time()
  #processing images
  '''
  Tiền xử lí ảnh, numpy() => muon dua anh vao 1 model AI thi minh phai convert no ve kieu cua model, tensor()
  '''
  im0 = letterbox(img_detect, img_size)[0]
  im0 = im0[:, :, ::-1].transpose(2, 0, 1)
  im0 = np.ascontiguousarray(im0)
  im0 = torch.from_numpy(im0).to(device)
  im0 = im0.half() if half else im0.float()
  im0 /= 255.0  # 0 - 255 to 0.0 - 1.0 RGB chuan hoa cac diem anh ve 0->1
  if im0.ndimension() == 3:
    im0 = im0.unsqueeze(0)
  # Inference
  t1 = time.time()

  #bat dau model pre
  pred = model(im0, augment= augment)[0]
  #output
  # print('pred : ')
  # print(pred)
  t2 = time.time()
  print('time detect : ', t2 - t1)
  # Apply NMS
  classes = None
  pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
  # Apply Classifier 
  if classify:
    pred = apply_classifier(pred, model, im0, img_ocr)
  gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
  points = []
  print('pred : ')
  print(pred)
  # [tensor([[175.62500,  61.00000, 201.87500,  87.87500,   0.92627,   5.00000], [x1, y1, x2, y2, conf, class]
  #       [ 68.12500,  64.31250, 125.62500, 111.43750,   0.92627,   5.00000],
  #       [ 11.81250, 104.00000, 123.93750, 221.00000,   0.89746,   3.00000],
  #       [144.50000,  63.12500, 167.00000,  81.37500,   0.87061,   5.00000],
  #       [184.75000,  75.37500, 232.25000, 151.62500,   0.85156,   3.00000],
  #       [129.50000,  87.75000, 183.00000, 133.50000,   0.81348,   3.00000]], device='cuda:0')]
  if len(pred[0]):
    check = True
    pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
    for c in pred[0][:, -1].unique():
      n = (pred[0][:, -1] == c).sum()  # detections per class
    for box in pred[0]:
      c1 = (int(box[0]), int(box[1]))
      c2 = (int(box[2]), int(box[3]))
      x1, y1 = c1
      x2, y2 = c2
      acc = round(float(box[4])*100,2)
      cls = int(box[5])
      conf = box[4].item()
      label = names[cls]
      cv2.rectangle(img_detect, c1, c2, (240,248,255), 2)
      cv2.putText(img_detect, label + ' ' + str(acc), c1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

      cv2.imwrite('img_detect.jpg', img_detect)
if __name__ == '__main__':
	use_gpu = torch.cuda.is_available()
	print('use_gpu : ',use_gpu)
	img_size = 416
	conf_thres = 0.25
	iou_thres = 0.45
	device = ''
	update = True
	# Load model yolo
	print('=================Loading models yolov5=================')
	t1 = time.time()
  #check co GPU ko 
	device = select_device(device)
	half = device.type != 'cpu'  # half precision only supported on CUDA
	# Load model nhan dien container
	t1 = time.time()
  #path file trong so
	weights = '/content/gdrive/MyDrive/v_train_yolo/PPE detect/yolov5/runs/train/200/weights/best.pt'
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	names = model.module.names if hasattr(model, 'module') else model.names
	print('names : ', names)
	if half:
		model.half()
	t2 = time.time()
	print('time load model yolo : ', t2-t1)

	#path file image test
	frame = cv2.imread('/content/gdrive/MyDrive/v_train_yolo/PPE detect/yolov5/data/images/TESTTT/test/images/000076_jpg.rf.49Lhm2iKLR6x3nY59RlL.jpg')
	#dau ra la anh da detect
	frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 320)

	# cap = cv2.VideoCapture('./data/filename_1thread.avi')
	# # count_frame = 0
	# try:
	# 	while(cap.isOpened()):
	# 		# Capture frame-by-frame
	# 		ret, frame = cap.read()
	# 		if ret == True:
	# 			# Display the resulting frame
	# 			if True:
	# 				t7 = time.time()
	# 				# frame = cv2.resize(frame, (640,640)) 
	# 				frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 320)
	# 				t8 = time.time()
	# 				print('total time :', t8 - t7)
	# 				count+= 1
	# 				if count % 100==0:
	# 					print(count)
	# 				if cv2.waitKey(25) & 0xFF == ord('q'):
	# 					break
	# 		else:
	# 			break
	# except KeyboardInterrupt:
	# 	cv2.destroyAllWindows()
	# 	cap.release()