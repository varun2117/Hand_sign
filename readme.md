### README: Hand Gesture Recognition System

---

#### **Overview**
This project implements a **Hand Gesture Recognition System** using computer vision and deep learning. The system captures hand gestures in real-time, processes them with Mediapipe, and classifies them into predefined categories using a trained PyTorch model.

---

#### **Features**
- **Real-Time Hand Gesture Detection**: Uses Mediapipe to extract hand landmarks from live video.
- **Customizable Dataset**: Collects and processes gesture data for training.
- **Deep Learning Model**: Implements a PyTorch-based neural network for gesture classification.
- **Interactive Inference**: Displays bounding boxes and predicted gestures live on the webcam feed.

---

#### **Requirements**
- Python >= 3.7
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `torch`
  - `scikit-learn`
  - `pickle`

Install the dependencies using pip:
```bash
pip install opencv-python mediapipe numpy torch scikit-learn
```

---

#### **Project Structure**
- **Data Collection**: Captures gesture data and saves it as labeled images.
- **Data Processing**: Extracts normalized hand landmarks and stores them in a structured format.
- **Model Training**: Trains a PyTorch model on the processed data.
- **Model Testing**: Evaluates the model’s performance on unseen data.
- **Live Inference**: Runs real-time gesture detection and classification.

---

#### **Usage Instructions**

##### **1. Data Collection**
- Start the data collection script:
  ```python
  python data_collection.py
  ```
- Use the webcam to capture gestures for each class. Press **'Q'** when ready to start capturing data.
- Each class’s data is stored in a separate folder within `./data`.

##### **2. Data Processing**
- Run the data processing script:
  ```python
  python data_processing.py
  ```
- This extracts hand landmarks using Mediapipe, normalizes them, and saves the data in `data.pickle`.

##### **3. Model Training**
- Train the gesture classification model:
  ```python
  python model_training.py
  ```
- The trained model is saved as `model.pth`.

##### **4. Real-Time Inference**
- Run the live inference script:
  ```python
  python live_inference.py
  ```
- The script displays the webcam feed with predictions and bounding boxes for recognized gestures. Press **'Q'** to quit.

---

#### **Customization**
- **Number of Classes**: Adjust the `number_of_classes` variable in the data collection script.
- **Dataset Size**: Modify `dataset_size` to control the number of samples per class.
- **Model Parameters**: Change the input size, hidden size, and output size in the model training script to suit your dataset.

---

#### **Key Components**
1. **Mediapipe Hands**: Detects hand landmarks and connections.
2. **PyTorch Model**: A simple feedforward neural network for gesture classification.
3. **Label Mapping**: Customize labels for gestures in the `labels_dict` dictionary.

---

#### **Example Output**
- Real-time webcam feed with:
  - **Bounding Box**: Highlights detected hands.
  - **Predicted Gesture**: Displays the recognized gesture above the bounding box.

---

#### **Challenges and Improvements**
- **Lighting and Backgrounds**: Train the model with diverse lighting conditions and backgrounds for robustness.
- **Class Overlap**: Ensure gestures are visually distinct for better accuracy.
- **Additional Classes**: Expand the dataset to include more gestures.

---

#### **Acknowledgments**
This project leverages:
- **Mediapipe** for efficient hand tracking and landmark detection.
- **PyTorch** for building and training the neural network.

---

#### **License**
This project is open-source under the MIT License. Contributions and improvements are welcome!

---

#### **Contact**
For questions or contributions, please open an issue or pull request on GitHub.
