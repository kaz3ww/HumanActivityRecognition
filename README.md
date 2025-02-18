---

# **Human Activity Recognition (HAR) using OpenCV and Deep Learning**

This project demonstrates how to recognize and classify human activities from video data using **OpenCV** and **Deep Learning**. The system captures motion in video streams and uses a deep learning model to classify activities. The model is trained using video data, and the system is capable of processing real-time video feeds for live activity recognition.

## **Technologies Used:**

- **Python**: The primary programming language used.
- **OpenCV**: For video and image processing.
- **TensorFlow / Keras**: For building and training deep learning models.
- **NumPy**, **Pandas**: For data manipulation.

---

## **Project Overview:**

This project is divided into three main parts:

### 1. **Data Collection**
   - Collects and preprocesses video data for activity recognition.
   
### 2. **Model Training**
   - Trains a **Convolutional Neural Network (CNN)** model for classifying human activities.
   
### 3. **Real-time Activity Recognition**
   - Uses the trained model to recognize human activities from a live video feed.

---

## **File Descriptions:**

### **1. `datacollection.py`**
   - This script captures video data and processes it into a format suitable for training.
   - Records the activities being performed (walking, sitting, jumping) and saves them for future use.

### **2. `modeltraining.py`**
   - Trains a CNN model for activity recognition.
   - Uses the data collected from `datacollection.py` to train the model.
   - Evaluates the model's performance and saves the trained model for future use.

### **3. `realtime.py`**
   - Runs a live video feed and applies the trained model to recognize human activities in real-time.
   - Uses the trained model to predict activities based on the current video frames.

---

## **Getting Started:**

### **1. Installation**

Ensure you have **Python 3.6** or higher installed. Then, install the required libraries using **pip**:

```bash
pip install opencv-python tensorflow numpy pandas
```

For visualizations (optional), you may also want to install **Matplotlib** and **Seaborn**:

```bash
pip install matplotlib seaborn
```

### **2. Data Collection**

Before training the model, you need to collect video data containing different activities.

#### Steps:
1. Open `datacollection.py` and run it to start recording.
2. The script will prompt you to perform the following activities:
   - **Walking**
   - **Sitting**
   - **Jumping**
   
   **Ensure to label each activity correctly when prompted.**
   
3. Press **'q'** to stop recording once you have enough data.

This will create a folder containing the image data labeled according to the activity you performed.

#### To record data:

```bash
python datacollection.py
```

---

### **3. Model Training**

Once you have collected sufficient data, it's time to train the model.

#### Steps:
1. Open `modeltraining.py` and ensure the paths to the training data are set correctly.
2. The script will load the collected data and train the CNN model using TensorFlow/Keras.

Training may take time depending on the dataset size. Once training is complete, the model will be saved for later use.

#### To train the model:

```bash
python modeltraining.py
```

After training, the model will be saved as **`activity_recognition_model.h5`** (or another name as set in the script).

---

### **4. Real-Time Activity Recognition**

Now that you have a trained model, you can run the system on a live video feed to classify activities in real-time.

#### Steps:
1. Open `realtime.py` and ensure the path to the saved model (`activity_recognition_model.h5`) is correctly specified.
2. The script uses your webcam (or any connected video feed) to detect and classify activities in real-time.

#### To run real-time recognition:

```bash
python realtime.py
```

You will see the predictions on the live video feed.

---

## **How to Record Data:**

1. **Run `datacollection.py`**.
2. **Perform Activities**: The script will ask you to perform activities such as **Walking**, **Sitting**, and **Jumping**.
3. **Label Your Activities**: As you perform each activity, the script will label the frames with the corresponding activity name.
4. **Stop Recording**: Once you have enough data, press 'q' to stop the recording.

---

### **Important Notes for Data Collection:**

- Ensure you have **clear visibility** of yourself when performing each activity.
- Perform each activity for **10-15 seconds** to gather enough frames for training.
- You can collect data for **multiple activities**, and more data will improve the model’s performance.

---

## **Troubleshooting:**

### **Q: The training is taking too long, what can I do?**
   - **A**: You can try using a smaller dataset for quicker results, or use a machine with more computational power (e.g., GPU).

### **Q: The model is not detecting activities correctly in real-time.**
   - **A**: Make sure the model is trained properly. Check that you have a **sufficiently large dataset** for training. Also, ensure the **webcam quality** is good enough for accurate predictions.

---

## **Contributing:**

If you would like to contribute to this project, feel free to fork the repository, make improvements, and submit pull requests. Contributions to improve the dataset, model accuracy, or real-time performance are welcome!

---

## **License:**

This project is open source and available under the **MIT License**.

---

### **Additional Resources:**

For further information on **Human Activity Recognition (HAR)** and **Deep Learning**, check out the following resources:

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---
