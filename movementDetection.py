import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

def checkHorizontalPosition(image, results, draw=False, display=False):
  horizontalPosition = None
  height, width, _ = image.shape
  outputImage = image.copy()

  left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
  right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)

  if (right_x <= width //2 and left_x <= width //2):
    horizontalPosition = 'Left'

  elif (right_x >= width //2 and left_x >= width //2):
    horizontalPosition = 'Right'

  elif (right_x >= width //2 and left_x <= width //2):
    horizontalPosition = 'Center'

  if draw:
    # Write the horizontal position of the person on the image. 
    cv2.putText(outputImage, horizontalPosition, (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Draw a line at the center of the image.
    cv2.line(outputImage, (width//2, 0), (width//2, height), (255, 255, 255), 2)


  if display:
    plt.figure(figsize=[10,10])
    plt.imshow(outputImage[:, :, ::-1]);plt.title('Output Image');plt.axis('off')

  else:
    return outputImage, horizontalPosition

def checkVerticalPosition(image, results, draw=False, display=False):
  height, width, _ = image.shape
  outputImage = image.copy()

  left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

  right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

  mid_y = right_y + left_y // 255

  upperBound = 200
  lowerBound = 280

  if (mid_y < upperBound):
    posture = 'Jumping'

  elif(mid_y > lowerBound):
    posture = 'Crouching'

  else: 
    posture = 'Standing'

  if draw:
    cv2.putText(outputImage, posture, (5, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.line(outputImage, (0, 240),(width, 240),(255, 255, 255), 2)

  if display:

    # Display the output image.
    plt.figure(figsize=[10,10])
    plt.imshow(outputImage[:,:,::-1]);plt.title("Output Image");plt.axis('off')
  
  else:  
    return outputImage, posture

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.flip(image, 1)

    image_height, image_width, _ = image.shape 
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS)
        
    if results.pose_landmarks:
      image, _ = checkHorizontalPosition(image, results, draw=True)
      image, _ = checkVerticalPosition(image, results, draw=True)

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
cap.release()


  