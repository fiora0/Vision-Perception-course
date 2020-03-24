### Vision and Perception course at the time of coronavirus

Face to face classes at this time are suspended and we are trying to help the students with online lectures, google classroms and video lectures.

In this repository there are examples, exercises and python implementation of algorithms for the course.

The course is divided in three parts: 
1) Multiview Geometry (Hartley & Zissermann), 2) Computer Vision Algorithm (Richard Szeliski) 3) Deep Learning for Computer Vision (Goodfellow & Bengio &Courville: Deep Learning Part III)

The program of the course can be found here:
[Program of Vision & Perception](https://sites.google.com/a/diag.uniroma1.it/visiope/home/program) 

Meeting is every Wednesday on Google Classroom
Video lectures are on YouTube


In the folder 

##  Removing projective distortion 

This is part of the online lecture of the Course of Vision and Perception of Wednesday 18 March 2020,


Simple implementation following the DLT algorithm as presented in Chapter 4 of  Hartley & Zisserman book 
Multiview geometry

also inspired by the matlab implementation of Peter Kovesi

##  Setup

Written in Python 3.6.7 


## Usage
clone or download the folder 

In the folder Dataset there are some images of buildings. Run the script transformation.py with a number between 0 and 6 
(there are 7 images in the dataset) an image is loaded, ensure that you have a folder named "dataset".

The image is shown and you should pick four points of a quadrilateral as in this example:
![Choose four points](https://github.com/fiora0/Vision-Perception-course/tree/master/removing_projective_distortion/choosefourpoints.png)


This  will be then mapped to a rectangle as shown in the following image showing your chosen four points (upsidedown) and the rectangle obtained from them.

![Rectangle](https://github.com/fiora0/Vision-Perception-course/tree/master/removing_projective_distortion/rectangle.png)

In the end you will se the image with removed projected distortion as this:

![Distortion removed](https://github.com/fiora0/Vision-Perception-course/tree/master/removing_projective_distortion/dist_removed.png)

You can add any other image you like to run the script. Be aware that the more distorted is the image the smaller will be the recovered building image and it might even disappear.



