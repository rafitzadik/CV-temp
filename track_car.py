#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 08:58:12 2017

@author: rafi
"""

from openalpr import Alpr
import random
import _init_paths
import cv2
import sys
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment # needed for the assignment problem
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CAR_CLASS = 7

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

class Car:
    def __init__(self):
        self.region = None
        self.pts = []
        self.new_pts = []
        self.index = None
        self.plate = None
        self.plate_conf = 0
        self.name = None
        self.state = None
        self.desc = ''
        self.color = (0,0,0)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

def scale(frame):
    max_axis = 500
    
    rows, cols = frame.shape[:2]
    #print ('frame rows: ', rows, ' cols: ', cols)
    
    rows_scale = float(rows) / float(max_axis)
    cols_scale = float(cols) / float(max_axis)
    factor = 1.0 / max(rows_scale, cols_scale)
    #print 'factor: ', factor
    scaled_frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    #print ('frame rows: ', scaled_frame.shape[0], ' cols: ', scaled_frame.shape[1])
    return scaled_frame, factor

def best_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    #print ('entering vis_detections')
    inds = np.where(dets[:, -1] >= thresh)[0]
    #print 'sizeof inds:', len(inds)
    if len(inds) == 0:
        return []

    im = im[:, :, (2, 1, 0)]
    best = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        best.append( (score, bbox) )
    return best
    
def find_cars(net, im):
    """Detect a car in an image using pre-computed object proposals."""

    # Load the demo image
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
#    print ('Detection took {:.3f}s for '
#           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    
    cls = 'car'
    cls_ind = 7 # 7 is the index of 'car' in CLASSES
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return best_detections(im, cls, dets, thresh=CONF_THRESH)
 
def unscale_regions(scaled_dets, scale_factor):
    unscaled = []
    for det in scaled_dets:
        sbbox = [int (float(x) / scale_factor) for x in det[1]]        
        unscaled.append( (det[0], sbbox) )
    return unscaled

def crop(frame, region):
    p1 = (int(region[1]), int(region[0]))
    p2 = (int(region[3]), int(region[2]))

    miny = min(p1[0], p2[0])
    maxy = max(p1[0], p2[0])
    minx = min(p1[1], p2[1])
    maxx = max(p1[1], p2[1])
    maxy = min(maxy, frame.shape[0])
    maxx = min(maxx, frame.shape[1])
    #print p1, p1, miny,maxy,minx,maxx
    c = frame[miny:maxy+1, minx:maxx+1]
    #print frame.shape, c.shape
    #cv2.imshow('crop', c)
    return c
    
def find_plate(img, alpr):
    ret,enc = cv2.imencode("*.bmp", img)
    results = alpr.recognize_array(bytes(bytearray(enc))) #is this how I get the frame to alpr?
    best_conf = 0.0
    best_plate = None
    for plate in results['results']:
        #print("   %12s %12s" % ("Plate", "Confidence"))
        for candidate in plate['candidates']:
#            prefix = "-"
#            if candidate['matches_template']:
#                prefix = "*"
            conf = candidate['confidence']
            p = candidate['plate']
            #print("  %s %12s%12f" % (prefix, p, conf))
            if (conf > best_conf and len(p) >= 6):
                best_conf = conf
                best_plate = p    
#    print 'detection for ', img.shape, ': ', best_plate
    return best_plate, best_conf

def find_plates(frame, alpr, cars):
    for car in cars:
        plate, conf = find_plate(crop(frame, car.region), alpr)
        if car.plate == None or conf > car.plate_conf:
            car.plate = plate
            car.plate_conf = conf
    
def draw_car(frame, car):
    r = car.region
    if (car.plate == None):
        text = car.name
    else:
        text = car.plate
    cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), car.color, 3)
    for pt in car.pts:
        cv2.circle(frame, (pt[0], pt[1]), 2, (0, 255, 0), -1)
    if (text):
        cv2.putText(frame, text, (r[0], r[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

def in_region(pt, r):
    (x,y) = pt
    return (x >= min (r[0], r[2])) and (x <= max(r[0], r[2])) and (y >= min(r[1], r[3])) and (y <= max(r[1], r[3]))

def car_pois(gray, region):
    mask = np.zeros_like(gray)
    assert(region[0]<region[2] and region[1] < region[3]) #that's my assumption, helps avoid taking min and max
    
    width = region[2] - region[0]
    height = region[3] - region[1]
    p0 = (int(region[0] + 0.1 * width), int(region[1] + 0.1 * height))
    p1 = (int(region[2] - 0.1 * width), int(region[3] - 0.1 * height))
    
    cv2.rectangle(mask, p0, p1, 255, -1)
#    cv2.imshow('gray', gray)
#    cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    p = cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)
    if (p == None):
        p = []
    #print "found ", len(p), " good features"
    pp = [[x,y] for [[x,y]] in p]
    return pp


def update_cars(cars, dets, prev_grey, cur_grey):
### Yes a third option, this time optimizing the matching of cars and dets using 
### a matching algorithm
    len_dets = len(dets)    
    dets_used = []
    if len(cars) > 0: #are we initialized at all?
        #build the cost matrix from current cars to current dets
        #the cost of matching car i to det region j would be the 
        #negative of the number of points in car i that
        #match a point in det region j
        #start by finding where each point matches
        prev_car_pts = []
        for car in cars:
            car.state = 'unmatched'
            car.new_pts = [] #delete the previous iteration
            for p in car.pts:
                prev_car_pts.append((car, p))
        p0 = np.float32([p for (car,p) in prev_car_pts])
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, p0, None, **lk_params)
        for (prv, cur, good) in zip(prev_car_pts, p1, st):
            if good == 1:
                prv[0].new_pts.append(cur)

        cost = np.zeros((len(cars), len(dets)))

        for i in range(len(cars)):
            for j in range(len(dets)):
                cost[i,j] = - len([p for p in cars[i].new_pts if in_region(p, dets[j][1])])
                
        car_ind, det_ind = linear_sum_assignment(cost)
        for (i,j) in zip(car_ind, det_ind):
            if cost[i,j] < 0: #hopefully not many where == 0
                cars[i].state = 'matched'
                cars[i].region = dets[j][1]
                cars[i].new_pts = [p for p in cars[i].new_pts if in_region(p, cars[i].region)]
                dets_used.append(j)
                
        #now just delete all unmatched cars
        cars = [car for car in cars if car.state == 'matched']
        #also, update each car that has too few poi's:
        for car in cars:
            if len(car.new_pts) < 30:
                car.new_pts = car_pois(cur_grey, car.region)
    #so we've taken care of all matches
    #delete all dets that are in some car
    dets = [dets[j] for j in range(len(dets)) if not j in dets_used]
    #make each remaining det a new car
    for det in dets:
        new_car = Car()
        new_car.region = det[1]
        new_car.state = 'new'
        new_car.name = 'Unrec' + str(random.randint(100,999))
        new_car.new_pts = car_pois(cur_grey, det[1])
        new_car.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cars.append(new_car)
    #finally push the new_pts to pts
    for car in cars:
        car.pts = car.new_pts
    assert(len(cars) == len_dets)        
    return cars

def process_frame(frame, net, alpr, cars, prev_grey):
    dets = find_cars(net, frame)
    cur_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = update_cars(cars, dets, prev_grey, cur_grey)
    find_plates(frame, alpr, cars)
    for car in cars:
        draw_car(frame, car)
    return cars, cur_grey
        
def process_video(vidFile, outFile, net, alpr):
    ret, frame = vidFile.read() # read first frame, and the return code of the function.
    num_frames = 0
    timer = Timer()
    cars = []
    prev_grey = None
    timer.tic()
    while ret:  # note that we don't have to use frame number here, we could read from a live written file.
        cars, prev_grey = process_frame(frame, net, alpr, cars, prev_grey)
        outFile.write(frame)
        cv2.imshow("frameWindow", frame)
        key = cv2.waitKey(1) # time to wait between frames, in mSec
        if (chr(key%256) == 'q'):
            ret = False
        else:
            ret, frame = vidFile.read() # read next frame, get next return code    
        num_frames += 1
    timer.toc()
    print num_frames, ' frames in ', timer.total_time, ' seconds'
    cv2.destroyAllWindows()
    key = cv2.waitKey(1)
    
def get_net():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = os.path.join(cfg.MODELS_DIR, NETS['zf'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS['zf'][1])
    print 'prototxt: ', prototxt
    print 'caffemodel: ', caffemodel
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net

def init_alpr():
    alpr = Alpr("us", "/etc/openalpr/openalpr.conf", 
                "/usr/share/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)
    
    alpr.set_top_n(20)
    return alpr
    
if __name__ == '__main__':    
    try:
        vidFile = cv2.VideoCapture('/home/rafi/Videos/park-small.mp4')
        #vidFile = cv2.VideoCapture(0)
    except:
        print "problem opening input stream"
        exit(1)
    if not vidFile.isOpened():
        print "capture stream not open"
        exit(1)

    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print "frame number: %s" %nFrames
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "FPS value: %s" %fps
    #fps = 25
    ret, frame = vidFile.read() #take one frame to get dimensions
    height = len(frame)
    width = len(frame[0])
    print "width: %s" %width
    print "height: %s" %height
    try:
        outFile = cv2.VideoWriter("/home/rafi/Videos/out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width,height), True)
    except:
        print "problem opening output stream"
        exit(1)
    if not outFile.isOpened():
        print "output stream not open"
        exit(1)
    
        
    print 'opened input and output video stream'
    net = get_net()
    print 'created caffe net'
    alpr = init_alpr()
    print 'opened ALPR'
    
    
    process_video(vidFile, outFile, net, alpr)
    outFile.release()
