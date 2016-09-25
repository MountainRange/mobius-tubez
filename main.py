import skvideo.io
import skvideo.datasets
import numpy as np

videogen = skvideo.io.vreader('test2.mp4')

print(videogen)

writer = skvideo.io.FFmpegWriter("outputvideo.mp4", {'-vsync':1})
for frame in videogen:
        writer.writeFrame(frame)
writer.close()