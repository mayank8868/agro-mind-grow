# Plant Disease Classification

This project implements a deep learning model for plant disease classification using PyTorch and EfficientNet-B0.

## Project Structure

```
backend/
├── data/                  # Data loading and processing
├── models/                # Saved models
├── utils/                 # Utility functions
├── train.py              # Training script
├── predict.py            # Prediction script
└── requirements.txt      # Dependencies
```

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Place your dataset in the following structure:
```
datasets/
└── plant_disease_recognition/
    └── train/
        ├── class1/
        │   ├── image1.jpg
        │   └── ...
        ├── class2/
        │   ├── image1.jpg
        │   └── ...
        └── ...
```

## Training

To train the model:
```bash
python train.py
```

Training parameters can be modified in the `Config` class within `train.py`.

## Prediction

To make predictions using a trained model:
```bash
python predict.py
```

Then enter the path to an image when prompted.

## Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Classifier**: Custom fully connected layer
- **Input Size**: 224x224 pixels
- **Augmentations**: Random crop, horizontal flip, rotation, color jitter

## Performance

The model achieves high accuracy on the plant disease classification task with proper training. The actual performance depends on the quality and quantity of training data.

## License

This project is licensed under the MIT License.
