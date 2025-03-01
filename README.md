# Foundations-of-Machine-Learning-Course-Final-Project
Foundations of machine learning (ML) course final project at University of Tehran (Fall 2024)

This project is about **voice signals analysis for authentication and gender classification**. Our notebook is publicly available [here](https://colab.research.google.com/drive/1Fv3pKzSQ3D0hdatKZLD7zGK9Ar4cm2FD).

## Contributors

- Paria Pasehvarz (Phase 1 & 2)
- Majid Faridfar (Phase 1 & 2)
- Fateme Mohammadi (Phase 1 & 2)
- Hannaneh Jamali (Phase 1)

## Phase 1 - Introduction to Voice Signals Analysis

This phase introduces fundamental concepts related to voice authentication and gender classification. The topics covered include:

- Overview of voice authentication methods, including closed-set and open-set authentication.
- Challenges in voice authentication and gender classification, along with potential solutions.
- Importance of preprocessing in voice authentication and gender classification, covering common steps such as noise reduction, normalization, and windowing.
- Feature extraction techniques such as Mel-Frequency Cepstral Coefficients (MFCC), Fast Fourier Transform (FFT), and log Mel spectrogram.
- Introduction to similarity learning and common loss functions such as contrastive loss and triplet loss.

## Phase 2 - Classification and Clustering

### Part 1 - Data Preprocessing

The dataset used for this study is available [here](https://drive.google.com/drive/folders/1pq_jGqdBda_QjNnK2yAzD4N2grbPF8Rs?usp=sharing). The preprocessing steps include:

- Selection of a balanced dataset with an equal number of male and female voice samples.
- Splitting the dataset into training (75%) and testing (25%) sets.
- Feature extraction, including duration, sampling rate, mean amplitude, max amplitude, min amplitude, spectral centroid, spectral bandwidth, zero-crossing rate, energy, log Mel spectrogram, and MFCC.
- Visualization of selected waveforms and spectrograms from the dataset.
- Noise reduction using spectral gating.
- Resampling audio files to a target sampling rate of 44,100 Hz to ensure consistency.
- Standardization of extracted features.
- Framing audio samples to a uniform frame size of 512 with a hop length of 512.

### Part 2 - Classification

Feature selection was performed to identify the most relevant features for gender classification. Methods such as Recursive Feature Elimination (RFE), Random Forest's Feature Importance, Mutual Information, and Principal Component Analysis (PCA) were explored.

The following models were used from sklearn library and evaluated using precision, recall, F1-score, confusion matrix, and ROC curve:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)

#### Task 1 - Gender Classification
Implemented classification models to determine gender from voice samples.

#### Task 2 - Closed-Set Authentication
For authentication, six students were randomly selected, and classification was performed three times with each model to evaluate consistency and performance.

### Part 3 - Clustering

Voice samples were clustered using:
- **KMeans Clustering**: Optimal `k` was determined using Silhouette Score and the Elbow Method.
- **DBSCAN Clustering**: Optimal parameters (`nearest neighbors` and `eps`) were determined using the Elbow Method.

PCA was used to visualize clusters in 2D and 3D. Clustering performance was evaluated using:
- Silhouette Score
- Davies-Bouldin Score
- Calinski-Harabasz Score

Results and analyses were conducted to assess the effectiveness of clustering techniques for voice classification.