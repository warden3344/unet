#!/bin/sh
cd /home/to/unet/
pwd
export PYTHONPATH=/path/to/unet:$PYTHONPATH
/home/to/caffe train --solver=solver.prototxt
