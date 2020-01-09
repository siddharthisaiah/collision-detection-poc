# Collision Detection - POC

This folder contains code to run collision detection between vehicles. Currently, it can detect collisions between

* car
* motorcycle
* bus
* truck

## Running the Code

- Place the video that you want to use as input in the `videos/` folder
- Edit the `video_name` variable in `object_detection/input_video.py` to match the filename
- Run the following commands
---
    $ cd object_detection/
    $ python collision_detection.py

## Known Issues

- Hatchbacks and Jeeps/SUVs take a longer time to be detected due to limitations of the training dataset used.
- Detection accuracy is reduced in poor lighting conditions.