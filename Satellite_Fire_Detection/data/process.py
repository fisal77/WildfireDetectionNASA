import numpy as np
from pyhdf.SD import SD
import os, sys
import skimage.io as io

a = os.listdir("./")
a.remove("process.py")
a.remove("nohup.out")

#img_, map_ = [], []

cnt = 0
for i in a:
    try:
        k = SD(i)
        k_img = k.select("MaxFRP").get()
        k_map = k.select("FireMask").get()

        for j in range(len(k_img)):
            io.imsave("MaxFRP/%d.jpg" % cnt, k_img[j])
            io.imsave("FireMask/%d.jpg" % cnt, k_map[j])
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
        os.remove(i)
    except:
        print(i)
        pass

#img_ = np.stack(img_)
#map_ = np.stack(map_)

#np.save("frp_img.npy", img_)
#np.save("fire_map.npy", map_)
