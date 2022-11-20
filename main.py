import numpy as np
import torch
import cv2

#
cap=cv2.VideoCapture("tvid.mp4")

# path='C:/Users/Shauvik/Desktop/FYProj/yolov5/yolov5s-int8.tflite'
#count=0


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

vehicles = [];

vehicles.append(model.names[2])
vehicles.append(model.names[3])
vehicles.append(model.names[5])
vehicles.append(model.names[7])

print("Testing")
print(vehicles)

# size=416

count=0
counter=0


color=(0,0,255)

cy1=250
offset=10
while True:
    ret,img=cap.read()

    count += 1
    if count % 4 != 0:
        continue
    img=cv2.resize(img,(600,500))
    cv2.line(img,(10,cy1),(599,cy1),(0,0,255),2)
    cntStr = f"Vehicle Count: {counter}"
    cv2.putText(img,str(cntStr),(80,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),1)

    res = model(img)
    for ind,row in res.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        d=row['name']
        v=''
        if(d=='motorcycle'):
            v = 'bike'
        else:
            v = d
        if d in vehicles:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
            mx1,my1 = ((x1+x2)/2,(y1+y2)/2)
            cen = int(mx1),int(my1)
            cv2.circle(img,(cen[0],cen[1]),2,(0,255,0),-1)
            cv2.putText(img,str(v),(x1,y1),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
            if cen[1] < (cy1+offset) and cen[1] > (cy1 - offset):
                counter +=1
                cv2.line(img,(50,cy1),(599,cy1),(0,255,0),2)
                print(cntStr)



    cv2.imshow("IMG",img)
    if cv2.waitKey(2)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
