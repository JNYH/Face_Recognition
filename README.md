# Face_Recognition
OpenCV python codes to recognise faces in images, webcam stream, and video files

Please feel free to fork this repo, and run the notebook. The dataset is in the sub-directories.


![image](https://user-images.githubusercontent.com/52286325/175816601-7420dd03-1d6f-4b67-b406-e9c83282127f.png)


## Face recognition with OpenCV, Python, and deep learning
Source: https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

In this tutorial, I have learnt how to perform facial recognition using OpenCV, Python, and deep learning. I started with a brief discussion of how deep learning-based facial recognition works, including the concept of “deep metric learning.” From there, I installed the libraries needed to perform face recognition. Finally, I implemented face recognition for both still images and video streams (such as webcam and video files). As discovered, the face recognition implementation will be capable of running in real-time.


![image](https://user-images.githubusercontent.com/52286325/175816746-c0fd0fa4-fec6-4fa5-a664-2ad466c6fee2.png)


In the `video_test` folder, the output files look something like this:

lunch_scene_output.avi https://youtu.be/MtBklF6ivmg

trailer_output.avi https://youtu.be/BxfdMrhsEnw


## Install face recognition libraries
In order to perform face recognition with Python and OpenCV, I will need to install these libraries:

The *dlib* library, maintained by Davis King, contains our implementation of “deep metric learning” which is used to construct our face embeddings used for the actual recognition process. Davis has provided a ResNet-based siamese network that is super useful for face recognition tasks. More details: https://pyimagesearch.com/2017/03/13/an-interview-with-davis-king-creator-of-the-dlib-toolkit/

The *face_recognition* library, created by Adam Geitgey, wraps around dlib’s facial recognition functionality, making it easier to work with. Adam’s library provides a wrapper around dlib to make the face recognition functionality easier to use. More details: https://adamgeitgey.com/

The *opencv* library, originally developed by Intel, is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. More details: https://pypi.org/project/opencv-python/

The *imutils* library, maintained by Adrian Rosebrock (pyimagesearch) is a series of convenience functions to make basic image processing functions easier with OpenCV. These processing functions include translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more. More details: https://github.com/PyImageSearch/imutils


## Encode the faces using OpenCV
Before being able to recognise faces in images and videos, I first need to encode (or quantify) the faces in my training set. Keep in mind that I am not actually training a network here - the network (in the library 'face_recognition') has already been trained to create 128-d embeddings from a dataset of ~3 million images.

I could alternatively train a network from scratch or even fine-tune the weights of an existing model, but that is too much to be done for many projects. Furthermore, I would need a lot of images to train the network from scratch. Instead, it is easier to use the pre-trained network and then use it to construct 128-d embeddings for each of the 30 faces in my dataset.

During classification, I have used a simple KNN model and votes to conclude the final face classification. Other traditional machine learning models could be used here as well. Please refer to **Face_Recognition.ipynb** for detailed code.


Here's an example webcam output:

![image](https://user-images.githubusercontent.com/52286325/175817262-82d9aa42-45b4-4d36-81ef-90285a2ba738.png)
