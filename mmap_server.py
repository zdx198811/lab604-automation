import mmap
#import shutil
from time import sleep

# Copy the example file
#shutil.copyfile('lorem.txt', 'lorem_copy.txt')


f = open('lorem_copy.txt', 'r+')
m = mmap.mmap(f.fileno(), 0,  flags=mmap.MAP_SHARED, )

for i in range(9999):
    #print('Before:\n{}'.format(m.readline().rstrip()))
    texttemp = str(i)
    m[0:11] = texttemp.rjust(11,'-').encode()
    m.seek(0)
    print('after: {}'.format(m.readline().rstrip()))  # rewind
    sleep(0.001)

m.close()
f.close()

