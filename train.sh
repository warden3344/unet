#!/bin/sh
cd /home/wangbin/test/unet/
pwd
export PYTHONPATH=/home/wangbin/test/unet:$PYTHONPATH
/home/wangbin/caffe-master/build/tools/caffe train --solver=solver.prototxt
