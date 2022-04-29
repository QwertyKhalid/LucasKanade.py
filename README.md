# LucasKanade.py
Optical flow with Lucas Kanade algorithm

The Lucas Kanade algorithm is a popular method used for achieving optical flow. In theory, a sequence of frames is given, and the intention is that between every pairs of frames, a vector is calculated that points from one pixel to the corresponding pixel in the next frame. In other words, the algorithms mainly tracks pixles between frames with the assumption that the displacement of the image content between the two adjacent images is small and is approximately constant. With the help of OpenCV, we can visualize optical flow with the Lucas Kanade algorithm in real time. Optical flow is used in for example self-driving cars to estimate movements on objects.
