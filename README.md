# Face_Recognition
OpenCV python codes to recognise faces in images, webcam stream, and video files


![image](https://user-images.githubusercontent.com/52286325/175816601-7420dd03-1d6f-4b67-b406-e9c83282127f.png)


## Face recognition with OpenCV, Python, and deep learning
Source: https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

Inside this tutorial, you will learn how to perform facial recognition using OpenCV, Python, and deep learning. We’ll start with a brief discussion of how deep learning-based facial recognition works, including the concept of “deep metric learning.” From there, I will help you install the libraries you need to actually perform face recognition. Finally, we’ll implement face recognition for both still images and video streams. As we’ll discover, our face recognition implementation will be capable of running in real-time.


![image](https://user-images.githubusercontent.com/52286325/175816746-c0fd0fa4-fec6-4fa5-a664-2ad466c6fee2.png)


Sample output videos:

lunch_scene output https://youtu.be/MtBklF6ivmg

trailer output https://youtu.be/BxfdMrhsEnw


## Install your face recognition libraries
In order to perform face recognition with Python and OpenCV, we need to install two additional libraries:

The dlib library, maintained by Davis King, contains our implementation of “deep metric learning” which is used to construct our face embeddings used for the actual recognition process. Davis has provided a ResNet-based siamese network that is super useful for face recognition tasks. More details: https://pyimagesearch.com/2017/03/13/an-interview-with-davis-king-creator-of-the-dlib-toolkit/

The face_recognition library, created by Adam Geitgey, wraps around dlib’s facial recognition functionality, making it easier to work with. Adam’s library provides a wrapper around dlib to make the face recognition functionality easier to use. More details: https://adamgeitgey.com/


![image](https://user-images.githubusercontent.com/52286325/175817262-82d9aa42-45b4-4d36-81ef-90285a2ba738.png)
