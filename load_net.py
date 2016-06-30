#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, './lib/models')
sys.path.insert(0, '/media/haria/data/py-faster-rcnn/lib')
sys.path.insert(0, '/media/haria/data/py-faster-rcnn/caffe-fast-rcnn/python')
import caffe
#from VGG16 import VGG16
from lib.models.faster_rcnn import FasterRCNN
import cPickle as pickle

from chainer import serializers

param_fn = './VGG16_faster_rcnn_final.caffemodel'
model_fn = './test.prototxt'

vgg = FasterRCNN()
net = caffe.Net(model_fn, param_fn, caffe.TEST)

for name, param in net.params.iteritems():
    print "====",name,"===="
    name = name.replace("/","_")
    if name not in ["rpn_cls_score","rpn_bbox_pred", "fc6", "fc7", "cls_score", "bbox_pred"]:
        trunk = vgg.trunk
        layer = getattr(trunk, name)
        layer.W = param[0].data
        layer.b = param[1].data
        setattr(trunk, name, layer)
    else:
        layer = getattr(vgg, name)
        layer.W = param[0].data
        layer.b = param[1].data
        setattr(vgg, name, layer)

serializers.save_npz('./VGG16_faster_rcnn_final.model', vgg)
