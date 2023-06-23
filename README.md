Real Time ASL Recognition System that can recognize American Sign Language from webcam or video in the real time with high accuracy.<br>
It consists of various models based on different algorithms like ann, cnn, cnn+transformer, lstm, transformer, etc.<br>
These models are trained in kaggle's asl dataset which contains data of 250 words.<br>
These models are taken from the Google - Isolated Sign Language Recognition Compition organized in kaggle.<br>
Competition: https://www.kaggle.com/competitions/asl-signs/overview<br>
Dataset: https://www.kaggle.com/competitions/asl-signs/data<br>


To run this project follow the steps bellow:

1. Make a Local Copy of the project.
2. Install the required dependency.
3. Run Each cell of 'Model Implementation' file.

Call "predict_asl" function to predict the asl from webcam or downloaded videos.

syntax: predict_asl(mode, video_path, model)

Option in predict_asl are:<br>
&nbsp;&nbsp;&nbsp;&nbsp; mode: 0-webcam & 1-local_video<br>
&nbsp;&nbsp;&nbsp;&nbsp; video_path: path of the video from local disk if mode=1 is selected<br>
&nbsp;&nbsp;&nbsp;&nbsp; model: ann, top-01, cnn, cnn+3trans, lstm, transformer and their is also default model if nothing is passed<br>

Owner details:<br>
&nbsp;&nbsp;&nbsp;&nbsp; Author: Satyaprakash Dewangan<br>
&nbsp;&nbsp;&nbsp;&nbsp; email: satyadewangan05@gmail.com<br>
&nbsp;&nbsp;&nbsp;&nbsp; linkedin: https://www.linkedin.com/in/satyadewangan/<br>
&nbsp;&nbsp;&nbsp;&nbsp; github: https://github.com/SatyaDewangan05<br>


Further Collaboration is most Welcome.
