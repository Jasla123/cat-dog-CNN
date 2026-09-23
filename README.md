# cat-dog-CNN

## Project Overview

This project applies Deep Learning techniques to classify images of cats and dogs using a Convolutional Neural Network (CNN).

A CNN model is built using TensorFlow/Keras to automatically learn visual features from images and classify them into two categories:

- Cat
- Dog

The project includes image preprocessing, data augmentation, CNN model development, model training, evaluation, and prediction on new images.

## Project Objective

The objective of this project is to develop a CNN-based image classification model that can distinguish between cats and dogs based on their visual features.

The model learns important image patterns such as shapes, edges, textures, and other visual characteristics during training.

## Dataset

The project uses a Cat and Dog image dataset containing separate training and testing images.

The dataset is organized into two classes:

- Cats
- Dogs

The images are loaded using Keras' "ImageDataGenerator" and "flow_from_directory()" methods.

The dataset structure is approximately:

data/
├── training_set/
│   └── training_set/
│       ├── cats/
│       └── dogs/
└── test_set/
    └── test_set/
        ├── cats/
        └── dogs/

The dataset itself is not included in this repository because image datasets can be large.

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the training and testing image datasets.
2. Resized images to 128 × 128 pixels.
3. Normalized pixel values to the range 0–1.
4. Applied data augmentation to the training images.
5. Used horizontal flipping, rotation, zoom, and shear transformations to improve model generalization.
6. Converted the images into batches for CNN training.

## Data Augmentation

Data augmentation was applied to the training dataset to generate variations of the existing images.

The following transformations were used:

- Rescaling
- Rotation
- Shearing
- Zooming
- Horizontal flipping

Data augmentation helps the model learn more general visual patterns and reduces the possibility of overfitting.

Convolutional Neural Network

A Convolutional Neural Network was developed using TensorFlow/Keras.

The model architecture includes:

- Conv2D layer
- MaxPooling2D layer
- Conv2D layer
- MaxPooling2D layer
- Flatten layer
- Dense layer
- Dropout layer
- Sigmoid output layer

The final sigmoid layer produces a probability used to classify the image as either a Cat or Dog.

## Model Training

The CNN model was trained using the prepared training dataset.

The model was compiled using:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy

The model learns to minimize classification error while improving its ability to distinguish between the two classes.

## Model Evaluation

The trained model was evaluated using the test dataset.

Test Performance

- Test Accuracy: 80.82%
- Test Loss: 0.4462

The training and validation performance were also visualized using accuracy and loss graphs to understand the learning behavior of the model.

## Prediction

The trained CNN model can be used to classify a new image.

An input image is:

1. Loaded using Keras.
2. Resized to 128 × 128 pixels.
3. Converted into an array.
4. Normalized.
5. Passed to the trained CNN model.
6. Classified as either Cat or Dog.

Example:

Input Image
     ↓
Resize (128 × 128)
     ↓
Normalize Pixel Values
     ↓
CNN Model
     ↓
Prediction
     ↓
Cat / Dog

Results

The CNN successfully learned to classify cat and dog images based on their visual features.

The model achieved approximately 80.82% accuracy on the test dataset.

The prediction results demonstrate that CNNs can be effectively used for binary image classification tasks.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL
- Jupyter Notebook

## Project Structure

cat-dog-cnn/
│
├── cat-dog-cnn.ipynb
├── README.md
├── .gitignore
│
└── sample/
    └── dog.jpg

## How to Run

1. Clone the Repository

git clone <your-repository-url>
cd cat-dog-cnn

2. Install Required Libraries

pip install tensorflow numpy matplotlib pillow

3. Add the Dataset

Place the dataset inside the "data/" directory following the structure described above.

4. Open the Notebook

Open:

cat-dog-cnn.ipynb

using Jupyter Notebook, JupyterLab, or Google Colab.

5. Run the Notebook

Run the cells from top to bottom to:

- Load the dataset
- Preprocess images
- Build the CNN
- Train the model
- Evaluate the model
- Make predictions

## Conclusion

This project demonstrates how Convolutional Neural Networks can be used for image classification.

The CNN learned visual patterns from cat and dog images and successfully classified unseen images into the two categories.

With further improvements such as a deeper CNN architecture, transfer learning, a larger dataset, and hyperparameter tuning, the classification performance could potentially be improved.

Future Improvements

Possible improvements include:

- Using Transfer Learning with models such as VGG16, ResNet, or MobileNet.
- Increasing the size and diversity of the dataset.
- Performing hyperparameter tuning.
- Adding more convolutional layers.
- Using early stopping and learning-rate scheduling.
- Creating a web application for real-time image prediction.
