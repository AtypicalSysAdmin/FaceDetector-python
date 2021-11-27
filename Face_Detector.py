import cv2 

#loads some pre-trained data on face frontals from open cv
trained_face_data= cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#choose an image to detect face in
# img= cv2.imread ('IMG_3332.JPG')
#getting default webcam video, args: index of the camera, 0 is the default cam, or name of a video file
cap = cv2.VideoCapture('Alison.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('FaceDetection.avi', fourcc, 20.0, size)

#loop forever in frames
while True:
    #read the current frame, returns true or false and the frame
    successful_frame_read, frame=cap.read()



    #convert images to black and white, args: src image and conversion of the color
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img, scaleFactor=1.1, minNeighbors=10)
    for (x, y, w, h) in face_coordinates:
        #draw the rectangle on the frame
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2 )
    # cv2.imshow('Window name', grayscaled_img)
    cv2.imshow('Recording...', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release the video capture object
cap.release()
out.release()
cv2.destroyAllWindows()

# Draw rectangles, args:  src img, coordinates, RGB color of the rectangle, the thickness of the rectangle



# print(face_coordinates)



# cv2.imshow('Clever programer', grayscaled_img)
# paueses until a key is pressed
# cv2.waitKey()



# print("Code ran completely")