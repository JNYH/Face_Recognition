from imutils import paths
import imutils, face_recognition, cv2, os, pickle, time
from collections import Counter

ti = time.time()
print('[INFO] creating facial embeddings...')
try:
    data = pickle.loads(open(os.getcwd() + '\\encodings.pickle', 'rb').read())    #encodings here
except FileNotFoundError:
    knownEncodings, knownNames = [], []
    imagePaths = list(paths.list_images(os.getcwd() + '\\dataset'))    #dataset here
    for (i, imagePath) in enumerate(imagePaths):
        print('{}/{}'.format(i+1, len(imagePaths)), end=', ')
        image, name = cv2.imread(imagePath), imagePath.split(os.path.sep)[-2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,  model='cnn')    #detection_method here
        for encoding in face_recognition.face_encodings(rgb, boxes):
            knownEncodings.append(encoding)
            knownNames.append(name)
    data = {'encodings': knownEncodings, 'names': knownNames}
    f = open(os.getcwd() + '\\encodings.pickle', 'wb')
    f.write(pickle.dumps(data))
    f.close()

print('Done! \n[INFO] recognising faces in video...')
stream = cv2.VideoCapture(os.getcwd() + '\\video_test\\trailer.mp4')    #test video here
writer = None

while True:
    (grabbed, frame) = stream.read()
    if not grabbed:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb, model='hog')    #detection_method here
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        votes = face_recognition.compare_faces(data['encodings'], encoding)
        if True in votes:
            names.append(Counter([name for name, vote in list(zip(data['names'], votes)) if vote == True]).most_common()[0][0])
        else:
            names.append('Unknown')
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top, right, bottom, left = int(top * r), int(right * r), int(bottom * r), int(left * r)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if writer is None:
        writer = cv2.VideoWriter(os.getcwd() + '\\video_test\\trailer_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    cv2.imshow('Video file', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
stream.release()
writer.release()
print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))
