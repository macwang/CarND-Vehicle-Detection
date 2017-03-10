#!/bin/sh

cd test_images
curl -C - -O https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip
curl -C - -O https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip
curl -C - -O https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
curl -C - -O https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

find ./ -name "*.zip" -exec unzip -q {} \;
