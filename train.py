import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import *

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

car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

car_image = cv2.imread(cars[car_ind])
car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
notcar_image = cv2.imread(cars[car_ind])
notcar_image = cv2.cvtColor(notcar_image, cv2.COLOR_BGR2RGB)

X = np.vstack((car))
