**Vehicle Detection Project**

It's the final project of term 1. The goals of this project is using traditional computer vision way to do vehicle detection rather than deep learning way. I also felt it's overkill to use deep learning because it will need more data and more time to train a working model.

The course provided a set of car/noncar 64x64 images. I started from examine some of them to decide how to extract features from it.

[//]: # (Image References)
[image1]: ./images/original_vs_smaller.png
[image2]: ./images/parameters.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Examine the provided car/noncar images

I usually check the distribution of the samples to see if it's balanced.
```python
cars = []
notcars = []

basedir = 'test_images/vehicles'
for d in os.listdir(basedir):
    cars.extend(glob.glob(basedir + '/' + d + '/*.png'))

basedir = 'test_images/non-vehicles'
for d in os.listdir(basedir):
    notcars.extend(glob.glob(basedir + '/' + d + '/*.png'))

print('%d car image found' % len(cars))
print('%d noncar image found' % len(notcars))
```

The result is 8792 and 8968. So good, the course has tailored a pretty good training dataset for us already.

## Perform HOG feature extraction

Then I randomly choose couple images and apply HOG to it. I tried dozen rounds, I noticed the HOG image of a car image must have a boundary around the car body. Like this

/------\
|      |
|      |
\------/

The HOG image of a noncar image is pretty random. It could be a combination of many parallel lines.

## Other features

Beside the HOG features, I also resized the 64x64 image to a smaller image. Ideally the larger image could have more details to be identified as a car. However, I can still identify it's a car from the shape and the color of tires even I downsampling it to 16x16.

![alt text][image1]

Although I felt it's quite strange to use color value distribution as a feature initially because there are at least hundreds of car colors. I'm sure the provided training samples didn't cover all of those. And not to mention the lighting condition will be vary. Then I convince myself that the goal of this project is for us to be familiar with these techniques, not to building an industrial-strength algorithm. And the project video is relatively simple. I also create the histogram of color channels and use it as part of the features.

So as a summary, I used 3 types of information as the features from the 64x64 training image.

1. flatten resized image(spatial_features, 3 channels x 16 in width x 16 in height = 768)
2. histogram of color channels(hist_features, 3 channels x 16 bins = 48)
3. HOG(3 channels x 7 Xcells x 7 Ycells x 2 Xblocks x 2 Yblocks x 9 orients = 5292)

There are 768 + 48 + 5292 = 6108 features.

## Training

There are many predefined classifier in the scikit-learn. I used linear SVM to train these data. Using the default parameters would have pretty good result in the end. I used all the provided data for training(17760 samples). I split it to two sets(80%/20%) for training and testing.

```python
clf = LinearSVC()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

The accuracy is 0.994.

## Testing different color spaces

I tried 48 different combinations of color spaces and HOG parameters. In the end I choose using YUV.

![alt text][image2]


===


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  