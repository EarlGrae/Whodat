this system is written to be run from command line.
most instals are straightforward. dlib is the tricky. 
if pip install dlib doesnt work, install cmake and append it's binaries to your path
then run pip install dlib --verbose
if that doesnt work message me
note dlib is a requirement of facial_recognition, even though it is not needed directly

encode images with encode_faces first. pass it:
the dataset folder with the -d tag
the encoding file with the -e tag
optionally either --detection-method cnn for good computers or
the lightweight --detection-method hog for less powerful computers, such as the pi
sample call:
python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

give it time to train. expect a good 30 mins.

recognition call:
pass it:
the cascade file. included. sample -c haarcascade_frontalface_default.xml
the assembled encodings with the -e tag
example call:
python pi_face_recognition.py -c haarcascade_frontalface_default.xml -e encodings.pickle
