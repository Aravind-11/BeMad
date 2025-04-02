# BeMaD: Behavioral Malware Detection

A reinforcement learning approach for network traffic malware detection using Deep Q-Networks (DQN).

## Overview

BeMaD is a project that leverages reinforcement learning techniques to detect malware in network traffic. The system uses Deep Q-Networks (DQN) to classify network flows into different malware categories based on their behavioral patterns.

## Features

- Deep Q-Network (DQN) implementation for malware classification
- Support for multi-class malware detection (5 classes)
- Data preprocessing and normalization for network traffic features
- Customizable neural network architecture
- Performance evaluation metrics

## Dataset

The project uses network traffic data with the following characteristics:
- Training set: 136,844 samples
- Test set: 32,583 samples
- 82 features (after preprocessing)
- 5 class labels (0-4) representing different types of network traffic

Class distribution:
```
Training set:
4    32608
3    31013
2    29792
1    25268
0    18163

Test set:
3    7754
2    7449
4    6522
1    6317
0    4541
```

## Requirements

```
tensorflow>=2.0.0
keras>=2.3.0
numpy>=1.21.1
pandas>=1.4.0
scikit-learn>=0.19.2
matplotlib>=3.3.3
tqdm>=4.25.0
```

A full list of dependencies can be found in `requirements.txt`.

## Project Structure

```
├── data/
│   ├── train_scaled.csv
│   └── test_scaled.csv
├── rl/
│   ├── agents/
│   │   └── dqn.py
│   ├── data.py
│   └── utils.py
├── models/
│   └── rl.pkl
├── train.py
├── train_dqn.ipynb
├── requirements.txt
└── README.md
```

## Usage

### Training the model

```python
from rl.agents.dqn import TrainDQN
from rl.data import get_train_test_val, load_csv
from tensorflow.keras.layers import Dense, Dropout
from keras.regularizers import l2, l1

# Load and preprocess data
X_train, y_train, X_test, y_test = load_csv("./data/train_scaled.csv", 
                                           "./data/test_scaled.csv", 
                                           "Label", 
                                           ["Flow ID", "Timestamp"], 
                                           normalization=False)

X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, 
                                                                    X_test, y_test,
                                                                    val_frac=0.3)

# Define model architecture
layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(5, activation=None)]  # 5 output classes

# Initialize DQN model
model = TrainDQN(episodes=60000, 
                warmup_steps=1700, 
                learning_rate=0.00025, 
                gamma=0.1, 
                min_epsilon=0.5, 
                decay_episodes=6000)

# Compile and train
model.compile_model(X_train, y_train, layers)
history = model.train(X_val, y_val, "Accuracy")

# Evaluate
stats = model.evaluate(X_test, y_test, X_train, y_train)
print(stats)
```

### Using the trained model

```python
import pickle

# Load the saved model
with open('models/rl.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
```

## Model Architecture

The default model architecture consists of:
- 3 dense layers with 256 neurons and ReLU activation
- Dropout layers (0.2) for regularization
- Output layer with 5 neurons (no activation) for Q-values

## Hyperparameters

Default hyperparameters:
- Episodes: 60,000 (or 150,000 in some experiments)
- Warmup steps: 1,700 (or 170,000 in some experiments)
- Learning rate: 0.00025
- Discount factor (gamma): 0.1
- Minimum epsilon: 0.5
- Batch size: 10,000 (or 32 in some experiments)
- Target network update period: 800
- Target update tau: 1.0
- N-step update: 1

## Results

The model evaluation provides metrics including accuracy, precision, recall, and F1-score for each class and overall performance.

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:
```
@software{bemad,
  author = {Aravind-11},
  title = {BeMaD: Behavioral Malware Detection},
  year = {2022},
  url = {https://github.com/Aravind-11/BeMaD}
}
```

## Acknowledgements

Special thanks to all contributors and researchers in the field of network security and reinforcement learning.
