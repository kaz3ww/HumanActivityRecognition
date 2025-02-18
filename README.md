# HumanActivityRecognition
This project implements Human Activity Recognition (HAR) using OpenCV and Deep Learning. It detects and classifies human activities in video data using motion detection and CNNs. Built with Python, OpenCV, and TensorFlow, it’s optimized for real-time analysis without external hardware.

Technologies Used:
Python
OpenCV for video and image processing
TensorFlow / Keras for building and training deep learning models
NumPy, Pandas for data manipulation
Project Overview:
This project is divided into three main parts:

Data Collection: Captures and preprocesses video data.
Model Training: Trains a deep learning model to classify human activities.
Real-time Activity Recognition: Uses the trained model to recognize human activities from a live video feed.
File Descriptions:
datacollection.py:

Collects video data for training.
Captures video frames and processes them into a format suitable for training the model.
Records the activities being performed and saves them for future use.
modeltraining.py:

Trains a Convolutional Neural Network (CNN) model for activity recognition.
Uses the preprocessed video data from datacollection.py to train the model.
Performs model evaluation and saves the trained model for future inference.
realtime.py:

Runs a live video feed and applies the trained model to recognize human activities in real time.
It uses the trained model to predict activities based on the current video frames.
Getting Started:
1. Installation:
Ensure you have Python installed (preferably Python 3.6 or higher). Then, install the required libraries using pip.

bash
Copy
Edit
pip install opencv-python tensorflow numpy pandas
You may also want to install Matplotlib and Seaborn for visualizations (optional):

bash
Copy
Edit
pip install matplotlib seaborn
2. Data Collection:
Before training the model, you need to collect video data containing different activities. Here's how to record and prepare the data:

Open the datacollection.py file and run it to start recording.
The script will prompt you to perform the following activities:
Walking
Sitting
Jumping
Make sure to clearly label each activity when prompted by the script.
The script saves the recorded video frames as image data, which will be used for model training.
To record data:

Run the datacollection.py script:
bash
Copy
Edit
python datacollection.py
Perform the activities (walking, sitting, jumping) when prompted.
Press 'q' to stop recording when done.
This will create a folder containing the image data labeled according to the activity you performed.

3. Model Training:
Once you have recorded enough data, it's time to train the model.

Open the modeltraining.py file and ensure the paths to the training data are correctly set.
The script will load the collected data and train a CNN model using TensorFlow/Keras.
This process can take time depending on the dataset size. The model will be saved once training is complete.
To train the model:

Run the modeltraining.py script:
bash
Copy
Edit
python modeltraining.py
After training, the model will be saved as activity_recognition_model.h5 (or another name as set in the script).
4. Real-Time Activity Recognition:
Now that you have a trained model, you can run the system on a live video feed to classify activities in real-time.

Open the realtime.py file and ensure the path to the saved model (activity_recognition_model.h5) is correctly specified.
The script uses your webcam (or any connected video feed) to detect and classify activities in real time.
To run real-time recognition:

Run the realtime.py script:
bash
Copy
Edit
python realtime.py
The system will use the webcam to classify activities, and you should see predictions on the live feed.
How to Record Data:
Run datacollection.py.
Perform Activities: The script will ask you to perform the following activities:
Walking
Sitting
Jumping
Label Your Activities: As you perform each activity, the script will label the frames you capture with the corresponding activity name.
Stop Recording: Once you have enough data, press 'q' to stop the recording.
Important Notes for Data Collection:
Ensure you have clear visibility of yourself when performing each activity.
Try to perform each activity for 10-15 seconds to get enough frames for training.
You can collect data for as many activities as you want. More data will improve model performance.
Troubleshooting:
Q: The training is taking too long, what can I do?

A: You can try using a smaller dataset for quicker results or use a machine with more computational power.
Q: The model is not detecting activities correctly in real-time.

A: Make sure the model is trained properly. Check that you are using a sufficiently large dataset for training. Also, ensure your webcam quality is good enough for accurate predictions.
Contributing:
If you would like to contribute to this project, feel free to fork the repository, make improvements, and submit pull requests. Contributions to improve the dataset, model accuracy, or real-time performance are welcome!

License:
This project is open source and available under the MIT License.
