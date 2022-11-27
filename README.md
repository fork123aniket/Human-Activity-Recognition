# Human Activity Recognition

This repository performs ***Human Activity Recognition***, where given a video, the developed framework recognizes the activity of the object per-frame using a pre-Trained 3D convolutional ResNet-34 model. The applications of it includes:-
- Automatically classifying/categorizing a dataset of videos on disk.
- Verifying that a food service worker has washed their hands after visiting the restroom or handling food that could cause cross-contamination (i.e,. chicken and salmonella).
- Monitoring bar/restaurant patrons and ensuring they are not over-served.

## Requirements
- `NumPy`
- `OpenCV`
- `collections`

## Usage
### Data
- The pre-Trained 3D ResNet-34 model of the developed framework is trained on the [***Kinetics dataset***](https://arxiv.org/abs/1705.06950).
- It consists of:-
  - 400 ***human activity recognition*** classes
  - At least 400 video clips per class (downloaded via YouTube)
  - A total of 300,000 videos
- The full list of classes the model can recognize can be seen [***here***](https://github.com/fork123aniket/Human-Activity-Recognition/blob/main/Class_Labels/action_recognition_kinetics.txt).
### Training and Testing
- The pre-Trained weights of the 3D convolutional ResNet-34 model, trained on the ***Kinetics dataset***, can be downloaded from [***here***](https://github.com/shuvamdas/human-activity-recognition/blob/master/resnet-34_kinetics.onnx).
- To see the implementation of ***Human Activity Recognition*** without ***rolling prediction***, check `Human_Activity_Recognition.py`. Moreover, to test this implementation on a test video, run the following command:-
`python Human_Activity_Recognition.py --model resnet-34_kinetics.onnx --classes Class_Labels/action_recognition_kinetics.txt --input example_activities.mp4`
- To see the implementation of ***Human Activity Recognition*** with ***rolling prediction***, check `Human_Activity_Recognition_Deque.py`. Besides, run the following command to check the performance of this implementation:-
`python Human_Activity_Recognition_Deque.py --model resnet-34_kinetics.onnx --classes Class_Labels/action_recognition_kinetics.txt --input example_activities.mp4`
## Results
Below are some results of developed model on test videos:-

| Test Video 1        | Test Video 2           | Test Video 3           |
| ------------------------- |:----------------------------:|:---------------------------:|
| ![alt text](https://github.com/fork123aniket/Human-Activity-Recognition/blob/main/Images/Sample1.gif) | ![alt text](https://github.com/fork123aniket/Human-Activity-Recognition/blob/main/Images/Sample2.gif) | ![alt text](https://github.com/fork123aniket/Human-Activity-Recognition/blob/main/Images/Sample3.gif) |

| Test Video 4        | Test Video 5           |
| ------------------------- |:----------------------------:|
| ![alt text](https://github.com/fork123aniket/Human-Activity-Recognition/blob/main/Images/Sample4.gif) | ![alt text](https://github.com/fork123aniket/Human-Activity-Recognition/blob/main/Images/Sample5.gif) |
