# Human Activity Recognition

This repository performs Human Activity Recognition, where given a video, the developed model recognizes the activity of the human per-frame. The applications of it includes:-
- Automatically classifying/categorizing a dataset of videos on disk.
- Verifying that a food service worker has washed their hands after visiting the restroom or handling food that could cause cross-contamination (i.e,. chicken and salmonella).
- Monitoring bar/restaurant patrons and ensuring they are not over-served.

## Requirements
- `NumPy`
- `OpenCV`
- `collections`

## Usage
### Data
- The developed model is trained on the [***Kinetics dataset***](https://arxiv.org/abs/1705.06950).
- It consists of:-
  - 400 human activity recognition classes
  - At least 400 video clips per class (downloaded via YouTube)
  - A total of 300,000 videos
### Training and Testing
- To see the implementation of Human Activity Recognition without ***rolling prediction***, check `Human_Activity_Recognition.py`. Moreover, to test this implementation on a test video, run the following command:-
`python Human_Activity_Recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4`
- To see the implementation of Human Activity Recognition with ***rolling prediction***, check `Human_Activity_Recognition_Deque.py`. Besides, run the following command to check the performance of this implementation:-
`python Human_Activity_Recognition_Deque.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4`
## Results
Below are some results of developed model on test videos:-
