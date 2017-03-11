import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

def process_image(image):
    return image

output = 'output_images/output.mp4'
clip1 = VideoFileClip('project_video.mp4')
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)
