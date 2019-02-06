import mmap
from time import sleep

# with open('lorem_copy.txt', 'r') as f:
    # with mmap.mmap(f.fileno(), 0,  flags=mmap.MAP_SHARED,
                   # access=mmap.ACCESS_READ) as m:
        # for i in range(10000):
            # sleep(0.1)
            # print(m[0:11])
for i in range(10000):
    sleep(0.1)
    with open('lorem_copy.txt', 'r') as f:
        with mmap.mmap(f.fileno(), 0,  flags=mmap.MAP_SHARED,
                       access=mmap.ACCESS_READ) as m:
            print(m[0:11])