import caffe

net = caffe.Net('train_val_test.prototxt', caffe.TEST)
print net.blobs['score'].data.shape
print net.blobs['crop5'].data.shape
