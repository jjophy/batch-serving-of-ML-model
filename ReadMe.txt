QIS Assignment part2 
Submitted by Jophy Joseph   Sep11 2022

In this assignment, we will build a docker container to perform batch serving of a ML
model. 
We need the following files:
● Dockerfile
● Train.py
● Inference.py
This is the EEG brainwave data that has been processed using statistical extraction.
There are totally 1300 rows and 162 columns in the train dataset. The feature Letter is
used as the target column. It has 26 classes which is a representation of the 26
alphabets.

Task:
train.py
1. Add one more model apart from the one used in TA Session (any machine
learning model of your choice) in the train.py file and save it. (5 points)
2. Modify the inference.py file to display the output of the above model. (5 points)
3. Build the docker image of the final application and run it and submit the
screenshot of the output. (15 points)
a. Build the Docker File
b. Run the docker container
c. Save the screenshot of the output and submit it.


My actions
1.) Have added a new model(Decision Tree Classifier) to the train.py
2.) dt.joblib file got created on execution of train.py
3.) On testing it with test.csv using inference.py, got expected results
4.) created new docker image with added model
5.) ran the new image and got expected results. 
6.) these images in the word file included
