import cv2
from crop import face_detection_crop
import math
import mediapipe as mp
def euclidean_distance(x1,y1,x2,y2):
    return   math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def eye_size(img_path):
    li=[]
    out_path=""
    _,_,w,h=face_detection_crop(img_path,img_path)
    img_temp=cv2.imread(img_path)
    img= cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB)
    face_mesh_mp=mp.solutions.face_mesh
    face_mesh=face_mesh_mp.FaceMesh(static_image_mode=True)
    face_point=face_mesh.process(img)
    if face_point.multi_face_landmarks:
        for landmark_list in face_point.multi_face_landmarks:
            l=[130,243,159,145,463,359,386,374]
            
            for i in l:
                li.append((landmark_list.landmark[i].x * w,landmark_list.landmark[i].y * h))

    
    right_side_width=euclidean_distance(li[0][0],li[0][1],li[1][0],li[1][1])
    right_side_height=euclidean_distance(li[2][0],li[2][1],li[3][0],li[3][1])
    left_side_width=euclidean_distance(li[4][0],li[4][1],li[5][0],li[5][1])
    left_side_height=euclidean_distance(li[6][0],li[6][1],li[7][0],li[7][1])
    
    for i in li:
        cv2.circle(img, (int(i[0]), int(i[1])), 5, (0, 255, 0), -1)
        cv2.imshow("Face Landmarks", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return right_side_width,right_side_height,left_side_width,left_side_height
def nose_lips_chin(img_path):
    li=[]
    out_path=""
    _,_,w,h=face_detection_crop(img_path,img_path)
    img_temp=cv2.imread(img_path)
    img= cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB)
    face_mesh_mp=mp.solutions.face_mesh
    face_mesh=face_mesh_mp.FaceMesh(static_image_mode=True)
    face_point=face_mesh.process(img)
    if face_point.multi_face_landmarks:
        for landmark_list in face_point.multi_face_landmarks:
            l=[168,14,152]
            
            for i in l:
                li.append((landmark_list.landmark[i].x * w,landmark_list.landmark[i].y * h))

    
    nose_lips=euclidean_distance(li[0][0],li[0][1],li[1][0],li[1][1])
    lips_chin=euclidean_distance(li[1][0],li[1][1],li[2][0],li[2][1])
    golden_ratio=nose_lips/lips_chin
    return nose_lips,lips_chin,golden_ratio
def forehead(img_path):
    li=[]
    
    _,_,w,h=face_detection_crop(img_path,img_path)
    img_temp=cv2.imread(img_path)
    img= cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB)
    face_mesh_mp=mp.solutions.face_mesh
    face_mesh=face_mesh_mp.FaceMesh(static_image_mode=True)
    face_point=face_mesh.process(img)
    if face_point.multi_face_landmarks:
        for landmark_list in face_point.multi_face_landmarks:
            l=[10,152,9]
            
            for i in l:
                li.append((landmark_list.landmark[i].x * w,landmark_list.landmark[i].y * h))

    
    top_bottom=euclidean_distance(li[0][0],li[0][1],li[1][0],li[1][1])
    point_10_9=euclidean_distance(li[0][0],li[0][1],li[2][0],li[2][1])
    forehead_size=h-top_bottom+point_10_9
    return forehead_size
    
    


   
if __name__ == "__main__":
    p1=""
    p2=""
    print(eye_size(p1))
    print(eye_size(p2))





