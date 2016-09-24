import skvideo.io
import skvideo.datasets
videodata = skvideo.io.vread('test1.mp4')
print(videodata.shape)