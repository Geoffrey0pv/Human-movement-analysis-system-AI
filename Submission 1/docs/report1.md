# First Delivery Report: Video Annotation System


**Course:** Algorithms and Programming III (APO3)

**Date:** 17/09/2025


**Team Members:**

* Raul Quigua
* Geoffrey Pasaje
* Mateo Rubio

---
## 1. Questions of Interest

The focus of this project is to answer the following questions through the development of a movement analysis system[cite: 69]:

* **Primary Question:** Is it possible to classify a person's activities (walking towards the camera, walking back, turning, sitting, standing up) with high accuracy and in real-time using joint data extracted from a video?
* **Secondary Questions:**
    * How can a person's lateral trunk tilt be measured to assess their postural stability? [cite: 30]
    * Which features extracted from joint movements (e.g., velocity, relative angles) are most influential for correctly predicting the performed activity? [cite: 35, 36, 37]
    * Which of the supervised classification models (SVM, Random Forest, XGBoost) offers the best performance for this specific task? [cite: 41]

---
## 2. Problem Type

This project addresses a **multiclass supervised classification problem on time-series data**.

* **Supervised Classification:** We will train a model using a pre-labeled dataset (videos annotated with the correct activity)[cite: 22, 41].
* **Multiclass:** There are more than two activities to classify (walking, turning, sitting, etc.)[cite: 13].
* **Time-Series:** The input is not a static image but a sequence of video frames. The classification of an activity depends on the evolution of joint positions over time.

---
## 3. Methodology

The work methodology will be based on the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** standard, adapted to the project's needs as suggested in the assignment guidelines[cite: 11]. The phases are as follows:

1.  **Business Understanding:** (Current Phase) Define the project objectives, questions of interest, and success criteria.

2.  **Data Understanding:** Collect the initial set of videos and perform an exploratory analysis to understand the variations between activities and individuals.

3.  **Data Preparation:**
    * Extract landmarks or key points of the joints from each frame using MediaPipe.
    * Normalize and filter the data to remove noise and make it scale-independent.
    * Generate relevant features (velocities, angles, trunk tilt).
    * Annotate and segment the data according to the performed activity.

4.  **Modeling:** Train and evaluate different classification models (SVM, Random Forest, etc.), adjusting their hyperparameters to find the best performance[cite: 10].
5. **Evaluation:** Measure the performance of the models on the test data using predefined metrics (precision, recall, F1-Score).
6. **Deployment:** Develop a simple graphical interface that displays the real-time classification and postural metrics.

---
## 4. Progress and Evaluation Metrics

To evaluate the performance of the activity classifier, we will use the following standard metrics:

* **Precision:** Of all the times the model predicted an activity (e.g., "sitting"), what percentage was correct? This is useful for knowing how reliable the model's predictions are.
* **Recall:** Of all the times a person actually performed an activity (e.g., "sitting"), what percentage did the model detect? This is key to avoid missing detections.
* **F1-Score:** The harmonic mean of Precision and Recall. It provides a single, balanced metric of performance, especially useful if the classes (activities) are imbalanced.
* **Confusion Matrix:** To visualize which activities the model confuses most often.

---

## 5. Data Collection and Exploratory Analysis

* **Collection Plan:** In this initial phase, no data has been collected yet. The plan is to record videos of 2-3 group members performing the 5 specified activities. We will aim to vary the speed, camera perspective, and movement trajectory to capture as much variability as possible.

* **Exploratory Analysis Plan:** Once the videos are collected, the exploratory analysis will consist of:
    1.  Processing a sample video with MediaPipe to extract joint coordinates.
    2.  Visualizing the "skeleton" overlay on the video to validate that the tracking is correct.
    3.  Plotting the trajectory of a key joint (e.g., the wrist) over time to observe patterns.
    4.  Calculating basic descriptive statistics on the joint positions.

---

## 6. Strategies to Augment the Dataset

Since collecting large volumes of video is a slow process, the following strategies are proposed to obtain more data if necessary:

1.  **Further Manual Collection:** Invite more volunteers to record videos performing the activities.
2.  **Data Augmentation:** Apply transformations to the existing videos to create new synthetic samples. Techniques include:
    * Horizontally flipping the video (simulates a right-handed person as left-handed and vice-versa).
    * Applying small variations in brightness, contrast, or noise to the video.
3.  **Search for Public Datasets:** Investigate the existence of publicly available databases containing videos of people performing similar actions that could be adapted for our problem.

---
## 7. Ethical Analysis

The implementation of this AI solution requires consideration of the following ethical aspects:

* **Privacy and Consent:** Informed consent must be obtained from all individuals being recorded. It must be guaranteed that the videos will be used exclusively for academic and research purposes and stored securely.
* **Data Bias:** If the dataset is collected solely from individuals with similar demographic characteristics (age, gender, body type), the model may not generalize well to other groups. It is important to seek diversity among participants.
* **Misuse of Technology:** Person-tracking technology could be used for surveillance purposes without consent. It is crucial to be aware of these implications and to define the project's scope strictly to movement analysis for sports, health, or human-computer interaction purposes.

---

## 8. Next Steps

The immediate steps to advance towards the second delivery (week 14) are:

1.  **Weeks 12-13:** Conduct the initial data collection (video recording).
2.  **Week 13:** Set up the development environment with the necessary libraries (OpenCV, MediaPipe).
3.  **Week 13:** Develop the initial scripts for data preprocessing: reading a video, extracting joint landmarks with MediaPipe, and saving them in a structured format (e.g., CSV).
4.  **Week 14:** Perform manual annotation of a subset of the data and begin training the first baseline model to obtain preliminary results.