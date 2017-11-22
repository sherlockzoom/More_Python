# coding=utf-8
"""
@created: 17-11-20
@author: zyl
"""
# import tensorflow as tf
#
# video_binay = tf.read_file('1509494430.mp4')
#
# with tf.Session() as sess:
#
#     video_form = tf.contrib.ffmpeg.decode_video(video_binay)
#     sess.run(video_form)

import numpy as np
import cv2

cap = cv2.VideoCapture(u'/home/zyl/PycharmProjects/More_Python/1509494430.mp4')

# fgbg = cv2.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    # fgmask = fgbg.apply(frame)
    if ret:
        cv2.imshow('frame',frame)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()