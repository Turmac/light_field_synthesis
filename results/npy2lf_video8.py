import os
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import os

output_video = True
display = True
display_grid = False

files = []
for idx in range(0, 4):
    files.append('output_%d.npy' % idx)

idx = 0
for f in files:
    if output_video:
        out = cv2.VideoWriter('video/ours-%d.mp4'%idx, cv2.VideoWriter_fourcc(*'mp4v'), 20, (528, 368))

    lf = np.load(f)
    box = 10
    
    img = None
    for i in range(8):
        for j in range(8):
            j2 = j if i%2 == 0 else 7-j

            frame = lf[0,:,:,i,j2,:]
            frame = np.clip(frame, 0, 1)

            x = i*box
            y = j2*box
            if display_grid:
                frame[0:10, 0:10] = 1
                frame[0:10, 70:80] = 1
                frame[70:80, 0:10] = 1
                frame[70:80, 70:80] = 1
                frame[x:x+box, y:y+box] = 0.5

            frame = frame[:368, :528]
            
            if display:
                if img is None:
                    img = plt.imshow(frame)
                else:
                    img.set_data(frame)
                    plt.pause(.01)
                    plt.draw()
            
            if output_video:
                frame = np.array(255*frame[...,::-1], dtype=np.uint8)
                out.write(frame)
    
    while i > 0:
        i -= 1
        frame = lf[0,:,:,i,j2,:]
        frame = np.clip(frame, 0, 1)

        x = i*box
        y = j2*box
        if display_grid:
            frame[0:10, 0:10] = 1
            frame[0:10, 70:80] = 1
            frame[70:80, 0:10] = 1
            frame[70:80, 70:80] = 1
            frame[x:x+box, y:y+box] = 0.5

        frame = frame[:368, :528]
            
        if display:
            if img is None:
                img = plt.imshow(frame)
            else:
                img.set_data(frame)
                plt.pause(.01)
                plt.draw()

        if output_video:
            frame = np.array(255*frame[...,::-1], dtype=np.uint8)
            out.write(frame)

    for i in range(1,8):
        for j in range(8):
            j2 = 7-j if i%2 == 0 else j
            frame = lf[0,:,:,j2,i,:]
            frame = np.clip(frame, 0, 1)

            x = j2*box
            y = i*box
            if display_grid:
                frame[0:10, 0:10] = 1
                frame[0:10, 70:80] = 1
                frame[70:80, 0:10] = 1
                frame[70:80, 70:80] = 1
                frame[x:x+box, y:y+box] = 0.5

            frame = frame[:368, :528]
            
            if display:
                if img is None:
                    img = plt.imshow(frame)
                else:
                    img.set_data(frame)
                    plt.pause(.01)
                    plt.draw()

            if output_video:
                frame = np.array(255*frame[...,::-1], dtype=np.uint8)
                out.write(frame)

    if output_video:
        out.release()
    idx += 1