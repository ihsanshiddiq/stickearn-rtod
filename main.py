import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture(0)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
cy1=345


tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
tracker4=Tracker()
tracker5=Tracker()



counter1=[]
counter2=[]
counter3=[]
counter4=[]
counter5=[]
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list1=[]
    motorcycle=[]
    list2=[]
    car=[]
    list3=[]
    person=[]
    list4=[]
    bus=[]
    list5=[]
    truck=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'motorcycle' in c:
            list1.append([x1,y1,x2,y2])
            motorcycle.append(c)
        elif 'car' in c:
            list2.append([x1,y1,x2,y2])
            car.append(c)
        elif 'person' in c:
            list3.append([x1,y1,x2,y2])
            person.append(c)
        elif 'bus' in c:
            list4.append([x1,y1,x2,y2])
            bus.append(c)
        elif 'truck' in c:
            list5.append([x1,y1,x2,y2])
            truck.append(c)
            
    bbox1_idx=tracker1.update(list1) # motorcycle
    bbox2_idx=tracker2.update(list2) # car
    bbox3_idx=tracker3.update(list3) # person
    bbox4_idx=tracker4.update(list4) # bus
    bbox5_idx=tracker5.update(list5) # truck

################## MOTORCYCLE #############################

    for bbox1 in bbox1_idx:
        for i in motorcycle:
            x3,y3,x4,y4,id1=bbox1
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
               if counter1.count(id1)==0:
                  counter1.append(id1)
   
################## CAR #############################

    for bbox2 in bbox2_idx:
        for i in car:
            x5,y5,x6,y6,id2=bbox2
            cxc=int(x5+x6)//2
            cyc=int(y5+y6)//2
            if cyc<(cy1+offset) and cyc>(cy1-offset):
               cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
               cv2.rectangle(frame,(x5,y5),(x6,y6),(0,0,255),1)
               cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
               if counter2.count(id2)==0:
                  counter2.append(id2)

################## PERSON #############################

    for bbox3 in bbox3_idx:
        for i in person:
            x7,y7,x8,y8,id3=bbox3
            cxc=int(x7+x8)//2
            cyc=int(y7+y8)//2
            if cyc<(cy1+offset) and cyc>(cy1-offset):
               cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
               cv2.rectangle(frame,(x7,y7),(x8,y8),(0,0,255),1)
               cvzone.putTextRect(frame,f'{id3}',(x7,y7),1,1)
               if counter3.count(id3)==0:
                  counter3.append(id3)

################## BUS #############################

    for bbox4 in bbox4_idx:
        for i in bus:
            x9,y9,x10,y10,id4=bbox4
            cxc=int(x9+x10)//2
            cyc=int(y9+y10)//2
            if cyc<(cy1+offset) and cyc>(cy1-offset):
               cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
               cv2.rectangle(frame,(x9,y9),(x10,y10),(0,0,255),1)
               cvzone.putTextRect(frame,f'{id4}',(x9,y9),1,1)
               if counter4.count(id4)==0:
                  counter4.append(id4)

################## TRUCK #############################

    for bbox5 in bbox5_idx:
        for i in truck:
            x11,y11,x12,y12,id5=bbox5
            cxc=int(x11+x12)//2
            cyc=int(y11+y12)//2
            if cyc<(cy1+offset) and cyc>(cy1-offset):
               cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
               cv2.rectangle(frame,(x11,y11),(x12,y12),(0,0,255),1)
               cvzone.putTextRect(frame,f'{id5}',(x11,y11),1,1)
               if counter5.count(id5)==0:
                  counter5.append(id5)


 

 ############### THE HOLY LINE ###############################
    cv2.line(frame,(0,cy1),(1018,cy1),(0,0,255),2)

  
    motorcyclec=(len(counter1))
    cvzone.putTextRect(frame,f'motorcycle-c: {motorcyclec}',(19,30),2,1)

    carc=(len(counter2))
    cvzone.putTextRect(frame,f'car-c: {carc}',(19,70),2,1)

    personc=(len(counter3))
    cvzone.putTextRect(frame,f'person-c: {personc}',(19,110),2,1)

    busc=(len(counter4))
    cvzone.putTextRect(frame,f'bus-c: {busc}',(19,150),2,1)

    truckc=(len(counter5))
    cvzone.putTextRect(frame,f'truck-c: {truckc}',(19,190),2,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Bus count : {busc}")
print(f"Truck count : {truckc}")
print(f"Car count : {carc}")
print(f"Motorcycle count : {motorcyclec}")
print(f"Person count : {personc}")
# print(motorcyclec)
# print(carc)




