import skimage.io as io
import numpy as np
import os

a = os.listdir("./FireMask/")
cnt = 0
for i in a:
    b = io.imread("./FireMask/" + i)
    b[b==1] = 0
    b[b==2] = 0
    b[b==6] = 0
    b[b==3] = 1 #non-fire water pixel
    b[b==4] = 2 #cloud (land or water)
    b[b==5] = 3 #non-fire land pixel
    b[b==7] = 4 #fire (low confidence, land or water)
    b[b==8] = 4 #Fire (nominal confidence, land or water)
    b[b==9] = 4 #Fire (high confidence, land or water)
    io.imsave("./FireMask/" + i, b)
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)
