import os
from os import listdir
from os.path import isfile, join


path = 'E:\\tmp'
files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]

i = 1
for f in files:
    print('processing %d\r' % i)
    os.system("lfptool raw -i %s --eslf-out" % (f))
    i += 1
print('finished')
