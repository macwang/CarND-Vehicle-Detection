**Vehicle Detection Project**

It's the final project of term 1. The goals of this project is using traditional computer vision way to do vehicle detection rather than deep learning way. I also felt it's overkill to use deep learning because it will need more data and more time to train a working model.

The course provided a set of car/noncar 64x64 images. I started from examine some of them to decide how to extract features from it.

[//]: # (Image References)
[image1]: ./images/original_vs_smaller.png
[image2]: ./images/parameters.png
[image3]: ./images/HOG_car.png
[image4]: ./images/heatmap.png
[image5]: ./images/p_test.png
[image6]: ./images/hog-compare.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Examine the provided car/noncar images

I usually check the distribution of the samples first to see if it's balanced.
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

Then I randomly choose couple images and apply HOG to it.

![alt text][image6]

I tried dozen rounds, I noticed the HOG image of a car image must have a boundary around the car body. Like this

![alt text][image3]

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

## Data preprocessing

It worth to mention that there are two actions I took before the training.

1. I only used OpenCV's imread() rather than matplotlib.image's imread() because it will produce array with value ranging 0 to 1 for PNG file. OpenCV's imread() will always produce array in 0 to 255. And I convert it from BGR to RGB.
2. Since the value is ranging from 0 to 255. It needs normalized. I used scikit-learn's StandardScaler to fit and transform the data before training.

```ptyhon
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```
The X_scaler has to be kept to normalize data when processing video.

## Sliding window search

Now I have a good classifier(99.2%) that can identify whether a 64x64 image block is car or not. So I will need to scan the whole image with sliding window to detect if it's a car. If it's detected as a car, then record the area. The detected rectangles might be overlapped. The more overlapped times area are more certain that's really a car. We can use heatmap to see those areas.

![alt text][image4]

## Optimization

The function to calculate HOG matrix took a lot of time. And the sliding window algorithm needs to scan hundreds times of the blocks. Many areas are redundantly calculated. So this place can be optimized.

I calculate the HOG matrix of the entire (400, 0)-(656, 1279) at once. And extracting the process area when I scan the image.

```python
# calulate HOG array
hog = get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False)

for ...     # Scan in X axis
    for ... # Scan in Y axis
        # extract the interesting area in the loop
        hog_features = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
```

## Preliminary test

![alt text][image5]

Here is the preliminary video. [YouTube](https://youtu.be/7gN-kQdQ86A)

The cars were roughly detected but the bounding boxes are very unstable. We need to use certain ways to make it move smoother and stabler.

## 
