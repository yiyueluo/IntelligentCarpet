import pickle
import h5py
import numpy as np
import os

#log_ = pickle.load(open("/home/shlee/aaSSD/MIT_data/test1/log.p", "rb"))

data_log = []
num = 0
for k in [70,80,90,100,110,120,130,140,160]:
    filename = "/home/shlee/aaSSD/MIT_data/MIT_test/ttyUSB0_"+str(k)+".hdf5"
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        pressure_data = f["pressure"]
        for i in range(0, pressure_data.__len__()):
            if (num % 200 == 0):
                data_log.append(num)
                os.mkdir("/home/shlee/aaSSD/MIT_data/test1/" + str(num // 200 * 200))

            d = [f["pressure"][i], np.array([k, 1])]
            pickle.dump(d, open("/home/shlee/aaSSD/MIT_data/test1/" + str(num // 200 * 200) + "/" + str(num) + '.p', "wb"))
            num+=1

        pickle.dump(data_log, open("/home/shlee/aaSSD/MIT_data/test1/" + 'log.p', "wb"))

#0.p : [ndarray(64,64), ndarray(1,2)]
print(data_log)