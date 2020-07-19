import cv2
import imutils
import numpy as np
from tensorflow.keras.models import model_from_json
import pyautogui

if __name__ == "__main__":

    json_file = open('8classes_32_32_3.json', 'r')
    loaded_json_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json_model)
    model.load_weights("8classes_32_32_3.h5")


    print("#####")


    camera = cv2.VideoCapture(0)

    classes = np.load("encoder_classes8.npy", allow_pickle = True).tolist()[0]
    #classes = ['No gesture','Stop Sign','Swiping Down','Swiping Left','Swiping Right','Swiping Up','Zooming In With Full Hand','Zooming Out With Full Hand']
    print(classes)
    num_frames = 0
    li = []
    top3 = [0,1,2]
    output = [0,0,0,0,0,0,0,0]
    label = "no action"
    cooldown = 0
    dic = {"Swiping Left":'left',"Swiping Right":'right'}
    
    while(True):

        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        #print(frame.shape)
        
        (height, width) = frame.shape[:2]

        right = int(width/2 + height/2)
        left = int(width/2 - height/2)
        top = 0
        bottom = height
        
        #inp = frame[top:bottom, left:right]
        inp = frame
        inp = cv2.resize(inp , (32,32))

        if num_frames < 36:
            li.append(inp)
        
        if num_frames >= 36:
            li.pop(0)
            li.append(inp)

        #print('c - ', cooldown , 'numf - ', num_frames)
        
        if num_frames > 35 and (num_frames-36)%18 == 0 and cooldown == 0 :
            inp_video = np.asarray(li)[None,:]
            output = model.predict(inp_video, verbose = 1, use_multiprocessing = 1)[0]
            index = np.argmax(output)
            top3 = output.argsort()[-3:]
            if index > 0:
                cooldown = 36
            label = classes[index]
            if "Swiping" in label:
                pyautogui.press(dic[label])
            print(label)
            
            
        
        if cooldown > 0:
            cooldown -= 1

        #cv2.rectangle(frame, (int(left), top), (int(right), bottom), (0,255,0), 2)
        num_frames += 1


        text = "activity: {}".format(label)
        #cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)


        backg = np.full((480, 1200, 3), 15, np.uint8)
        backg[:480, :640] = frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(backg, text,(40,40), font, 1,(0,0,0),2)
        for i, top in enumerate(top3):
            cv2.putText(backg, classes[top],(700,200-70*i), font, 1,(255,255,255),1)
            cv2.rectangle(backg,(700,225-70*i),(int(700+output[top]*170),205-70*i),(255,255,255),3)

        cv2.imshow('preview',backg)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
