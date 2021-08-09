import pickle
import h5py
import numpy as np
import os

log = pickle.load(open("/home/shlee/IntelligentCarpet/train/"+ 'log.p', "rb"))
data0 =  pickle.load(open("/home/shlee/IntelligentCarpet/train/"+ '0.p', "rb"))

filename = "/home/shlee/aaSSD/MIT_data/ttyUSB0.hdf5"

data_log = []
with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    pressure_data = f["pressure"]
    for i in range(0,pressure_data.__len__()):
        if (i%200 ==0):
            data_log.append(i)
            os.mkdir("/home/shlee/aaSSD/MIT_data/test/"+str(i//200*200))

        d = [f["pressure"][i], np.array([1,1])]
        pickle.dump(d, open("/home/shlee/aaSSD/MIT_data/test/"+str(i//200*200)+"/" + str(i) + '.p', "wb"))

    pickle.dump(data_log, open("/home/shlee/aaSSD/MIT_data/" +'log.p', "wb"))

#0.p : [ndarray(64,64), ndarray(1,2)]
print(data_log)