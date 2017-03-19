import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
# from utils import *

# Load the trained model, create clf

# Load the scaler

# Create windows

def process_image(image):
    # w = search_windows(img, windows, clf, scaler, color_space='RGB',
    #                    spatial_size=(32, 32), hist_bins=32,
    #                    hist_range=(0, 256), orient=9,
    #                    pix_per_cell=8, cell_per_block=2,
    #                    hog_channel=0, spatial_feat=True,
    #                    hist_feat=True, hog_feat=True):

    # 1. sliding window -> filter out postivie windows
    # 2. Choose the central of each overlapped areas
    # 3. false postive is filtered by looking previous frame
    # 4. Recording the central of each rect frame to frame
    # 5.

    return image

output = 'output_images/output.mp4'
clip1 = VideoFileClip('project_video.mp4')
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)
