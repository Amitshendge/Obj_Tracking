from dis import dis
import re
from turtle import distance
import cv2
import numpy as np
from object_detection import ObjectDetection
import obj_detection
import os
import math
#Initialize Object Detection
od=obj_detection.Obj_detect()
temp=True
#Split video into frame using OpenCV
cap= cv2.VideoCapture("sampleSuperTrim5.mp4")
# cap.set(cv2.CAP_PROP_FPS, 5)
naming=10000000
count=0
BinsIn=0
BinsOut=0
centre_points_prev_frame=[]
textfile=open("total_output.txt","a")

tracking_objects={}
tracking_distance={}
track_id=1
while True:
    ret,frame=cap.read()
    if ret==False:
        break
    hei,wid,nnnn=frame.shape
    # print(hei , wid)
    h1,h2,w1,w2=int(hei/1.75),int(hei/1.15),int(wid/4.2),int(wid/1.6)
    # roi=frame[h1:h2,w1:w2]

    print(count)
    count=count+1
    if not ret:
        break
    #points of current frame
    centre_points_cur_frame=[]
    cv2.imwrite(("frame.jpg"),frame)
    # Detect objects on frame
    (class_id,scores,boxes)=od.main()
    # os.remove("frame/frame.jpg")
    for box in boxes:
        (left,top,height,width)=box
        # left,top,height,width=left+w1,top+h1,height,width
        points=((left,top),(left+width,top),(left+width,top+height),(left,top+height),(left,top))
        cx=int((left+left+width)/2)
        cy=int((top+top+height)/2)
        centre_points_cur_frame.append((cx,cy))
        cv2.rectangle(frame,(left,top),(left+width,top+height),(0,255,0),2)
        # cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
    if count <=2:
        for pt in centre_points_cur_frame:
            # cv2.circle(frame,pt,5,(0,0,255),-1)
            for pt2 in centre_points_prev_frame:
                distance=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                if distance<50:
                    tracking_objects[track_id]=pt
                    try:
                        tracking_distance[track_id]=tracking_distance[track_id]+distance
                    except:
                        tracking_distance[track_id]=distance
                    track_id+=1
                    # mystr="[Obj id: "+str(track_id)+"]"+"[Curr: "+str(centre_points_cur_frame)+"][Prev: "+str(centre_points_prev_frame)+"][Distance: "+str(distance)+"]\n"
                    # print(mystr)
                    # textfile.write(mystr)
    else:
        if temp:
            tracking_distance.pop(1)
            temp=False
        tracking_objects_copy=tracking_objects.copy()
        centre_points_cur_frame_copy=centre_points_cur_frame.copy()

        for object_id,pt2 in tracking_objects_copy.items():
            object_exists=False
            for pt in centre_points_cur_frame_copy:
                distance=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                #update object position
                if distance<50:
                    tracking_objects[object_id]=pt
                    if pt[1]<pt2[1]:
                        distance=-distance
                    try:
                        tracking_distance[track_id]=tracking_distance[track_id]+distance
                    except:
                        tracking_distance[track_id]=distance
                    # mystr="[Obj id: "+str(track_id)+"]"+"[Curr: "+str(centre_points_cur_frame)+"][Prev: "+str(pt2)+"][Distance: "+str(distance)+"]\n"
                    # print(mystr)
                    # textfile.write(mystr)
                    object_exists=True
                    centre_points_prev_frame=centre_points_cur_frame.copy()
                    if pt in centre_points_cur_frame:
                        centre_points_cur_frame.remove(pt)
                    continue
            #remove lost id
            if not object_exists:
                if tracking_distance[object_id+1]>0:
                    BinsIn=BinsIn+1
                else:
                    BinsOut=BinsOut+1
                tracking_objects.pop(object_id)

        #Add new ID found
        for pt in centre_points_cur_frame:
            tracking_objects[track_id]=pt
            track_id+=1

    for object_id,pt in tracking_objects.items():
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
        cv2.putText(frame,str(object_id),(pt[0],pt[1]-7),0,1,(0,0,255),2)

    naming+=1
    cv2.putText(frame,"Total Bins: "+str(track_id-1),(50,50),0,1,(0,0,255),2)
    cv2.putText(frame,"Bins In: "+str(BinsIn),(50,100),0,1,(0,0,255),2)
    cv2.putText(frame,"Bins Out: "+str(BinsOut),(50,150),0,1,(0,0,255),2)
    cv2.imshow("Frame",frame)
    # cv2.imshow("roi",roi)
    cv2.imwrite(("Final/frame"+str(naming)+".jpg"),frame)
    #make a copy of points
    centre_points_prev_frame=centre_points_cur_frame.copy()
    key=cv2.waitKey(1)

    if key=='q':
        break
print(tracking_distance)
cap.release()
cv2.destroyAllWindows()
