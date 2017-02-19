#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 08:58:12 2017

@author: rafi
"""

from openalpr import Alpr
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


def scale(frame):
    max_axis = 500
    
    rows, cols = frame.shape[:2]
    #print ('frame rows: ', rows, ' cols: ', cols)
    
    rows_scale = float(rows) / float(max_axis)
    cols_scale = float(cols) / float(max_axis)
    #factor = 1.0 / max(rows_scale, cols_scale)
    factor = 1
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
        return

    im = im[:, :, (2, 1, 0)]
    best = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        best.append( (score, bbox) )
    return best
    
def find_car(net, im):
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
    print p1, p1, miny,maxy,minx,maxx
    c = frame[miny:maxy+1, minx:maxx+1]
    print frame.shape, c.shape
    #cv2.imshow('crop', c)
    return c
    
def find_plate(img, alpr):
    ret,enc = cv2.imencode("*.bmp", img)
    results = alpr.recognize_array(bytes(bytearray(enc))) #is this how I get the frame to alpr?
    best_conf = 0.0
    best_plate = 'None'
    for plate in results['results']:
        #print("   %12s %12s" % ("Plate", "Confidence"))
        for candidate in plate['candidates']:
#            prefix = "-"
#            if candidate['matches_template']:
#                prefix = "*"
            conf = candidate['confidence']
            p = candidate['plate']
            #print("  %s %12s%12f" % (prefix, p, conf))
            if (conf > best_conf):
                best_conf = conf
                best_plate = p    
    print 'detection for ', img.shape, ': ', best_plate
    return best_plate
    
def draw_region(frame, r, plate):
    cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 3)
    if (plate):
        cv2.putText(frame, plate, (r[0], r[3]), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

#def process_frame(frame, net, alpr):
# this version tries to deal with scaling
#        scaled_frame, scale_factor = scale(frame)
#        scaled_dets = find_car(net, scaled_frame)
#        if scaled_dets: # did we find anything?
#            dets = unscale_regions(scaled_dets, scale_factor)
#            print 'found ', len(dets), ' objects'
#            for det in dets:             
#                #plate = find_plate(frame, alpr)
#                plate = find_plate(crop(frame, det[1]), alpr)
#                #plate = 'None'
#                draw_region(frame, det[1], plate)
##                cv2.imshow(plate, crop(frame, det[1]))
##                key = cv2.waitKey(0) # time to wait between frames, in mSec

def process_frame(frame, net, alpr):
        look_for_plates = True
        dets = find_car(net, frame)
        if dets: # did we find anything?
            print 'found ', len(dets), ' objects'
            for det in dets:             
                if look_for_plates:
                    plate = find_plate(crop(frame, det[1]), alpr)
                else:
                    plate = '----'
                draw_region(frame, det[1], plate)
#                cv2.imshow(plate, crop(frame, det[1]))
#                key = cv2.waitKey(0) # time to wait between frames, in mSec
    
def process_video(vidFile, net, alpr):
    ret, frame = vidFile.read() # read first frame, and the return code of the function.
    num_frames = 0
    timer = Timer()
    timer.tic()
    while ret:  # note that we don't have to use frame number here, we could read from a live written file.
        process_frame(frame, net, alpr)
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
    
#def process_video(vidFile, net, alpr):
#    ret, frame = vidFile.read() # read first frame, and the return code of the function.
#    while ret:  # note that we don't have to use frame number here, we could read from a live written file.
#        scaled_frame, scale_factor = scale(frame)
#        scaled_dets = find_car(net, scaled_frame)
#        if scaled_dets: # did we find anything?
#            print 'found ', len(scaled_dets), ' objects'
#            for det in scaled_dets:             
#                #plate = find_plate(frame, alpr)
#                int_det = [det[0], [int(x) for x in det[1]]]
#                plate = find_plate(crop(scaled_frame, int_det[1]), alpr)
#                draw_region(scaled_frame, int_det[1], plate)
#                cv2.imshow("crop", crop(scaled_frame, int_det[1]))
#                key = cv2.waitKey(0) # time to wait between frames, in mSec
#                if (chr(key%256) == 'q'):
#                    ret = False
#        cv2.imshow("frameWindow", scaled_frame)
#        key = cv2.waitKey(1) # time to wait between frames, in mSec
#        if (chr(key%256) == 'q'):
#            ret = False
#        else:
#            ret, frame = vidFile.read() # read next frame, get next return code    

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
        vidFile = cv2.VideoCapture('/home/rafi/Videos/reverse park.mp4')
        #vidFile = cv2.VideoCapture(0)
    except:
        print "problem opening input stream"
        exit(1)
    if not vidFile.isOpened():
        print "capture stream not open"
        exit(1)

    print 'opened video stream'
    net = get_net()
    print 'created caffe net'
    alpr = init_alpr()
    print 'opened ALPR'
    
    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print "frame number: %s" %nFrames
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "FPS value: %s" %fps
    #fps = 25
    
    process_video(vidFile, net, alpr)
