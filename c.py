import cv2
import torch
import numpy as np
import telepot
from matplotlib import path
vid=cv2.VideoCapture("rtsp://admin:admin1234@192.168.1.15:554/cam/realmonitor?channel=1&subtype=0")
model=torch.hub.load('ultralytics/yolov5','custom',path='yolov5s.pt')
# model=torch.hub.load('yolov5','custom',path='yolov5s.pt')
model.classes=[0]
classes=model.names
device='cuda' if torch.cuda.is_available() else 'cpu'
def score_frame(frame):
    model.to(device)
    frame=[frame]
    results=model(frame)
    labels,cord=results.xyxyn[0][:,-1].numpy(),results.xyxyn[0][:,:-1].numpy()
    return labels,cord
def class_to_label(x):
    return classes[int(x)]
def Plot_boxes(results,frame):
    labels,cord=results
    print("labels",labels)
    print("cord",cord[:,:-1])
    clas=0
    if len(labels) !=0:
        print("list is not empty")
        for label in labels:
            if label==clas:
                print("send objects")
            else:
                print("wrong objects")
    else:
        print("list is empty")
        print("no objects")
    n=len(labels)
    x_shape,y_shape=frame.shape[1],frame.shape[0]
    for i in range(n):
        row=cord[i]
        # print("predict",round(cord[i][4],2))
        if row[4]>=0.2:
            x1,y1,x2,y2=int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
            x3=int(row[0]*x_shape)+ int(row[2]*x_shape)/2
            y3=int(row[1]*y_shape)+int(row[3]*y_shape)/2
            bgr=(0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
            cv2.putText(frame,class_to_label(labels[i])+" "+str(round(row[4],2)),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.9,bgr,2)
            cv2.rectangle(frame,(0,0),(800,500),(255,0,0),10)
            b=(0,0)[0]<x3<(800,500)[0] and (0,0)[1]<y3<(800,500)[1]
            if b:
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                token="6275415240:AAF3yDdT45-VIn8GdBrQUHH0XmtMXo0MC28"
                receiver_id=5877612764
                bot=telepot.Bot(token)
                # bot.sendMessage(receiver_id,"hai")
                # bot.sendPhoto(receiver_id,photo=open("1.png", "rb"), caption="Có xâm nhập, nguy hiêm!")
                bot.sendVideo(receiver_id,video=open("a.mp4", "rb"),supports_streaming=True)
                cv2.putText(frame, "Tinh trang: {}".format(text), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame
while True:
    ret,frame=vid.read()
    results=score_frame(frame)
    # print(results)
    frame=Plot_boxes(results,frame)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
