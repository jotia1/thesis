import numpy as np
import lmdb
import caffe
import loadaerdat


N = 1000 # This will need to correspond to #frames to produce
DVS_res = 128  # DVS camera resolution 
filename = 'samp5.aedat'

ts, xpos, ypos, pol = loadeardat(filename)

# Data will be one value, per DVS pixel from the past
X = np.zeros(N, 1, DVS_res, DVS_res)
# Labels will be all of the future 
Y = np.zeros(N, 1, DVS_res, DVS_res)

# TODO experiement with *10 ... why do we need...
env = lmdb.open('dvsDataLMDB', map_size=X.nbytes*10)

with env.begin(write=True) as db:

	for i in range(N):
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        #datum.data = X