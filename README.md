# EuroSAT-Land-Use-Classification
This project explores land-use classification using **EuroSAT RGB** dataset. The goal is to compare the performance of a machine learning algorithm with traditional feature extraction deep learning with automatically learned features.
* Random Forest: 513 features were extracted using OpenCV from each image. 512 of these features were the HSV color histogram bins which represent the various combinations of hue, saturation, and value. Finally, Canny edge density was added, bringing the total number to 513 features per image.
* Convolutional Neural Network (CNN): In this approach, the feature learning is done automatically. Instead of passing features extracted from images, the model takes the images as inputs and learns various features during training.

## Pipeline
- Setup: Loads the dataset and checks the GPU
- Explore and split the data: counts images from each class, and visualizes an instance. Then, loads the subsets and performs the initial train-test  split.
- Feature Extraction and Random Forest: Extracts HSV histogram and edge density, then trains a random forest.
- CNN: Converts the data to tensors, normalizes and holds out a validation set, then trains a small three-block CNN.
- Analysis: Evaluates each model using confusion matrices and F1 scores.

## Dataset
The EuroSAT RGB dataset consists of 27000 images, each measuring 64x64, divided across 10 classes. In this project, 1500 instances of each class was used. 80% of the data was used for training, with a similar proportion from each class in every batch.

| Class | Number of Images |
|---|---:|
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |
| **Total** | **27,000** |

<img width="1379" height="601" alt="image" src="https://github.com/user-attachments/assets/ad1ee051-f183-4ede-aa20-3b845828f96e" />


## Results

| Model | Test Accuracy |
|---|---:|
| OpenCV Features (HSV Histogram + Edge Density) + Random Forest | 81.5% |
| Small CNN (PyTorch) | 94.6% |

The largest improvements, with over 25%, were on Highway and River which were the weakest predictions of random forest.
| Class | Random Forest F1 | CNN F1 |
|---|---:|---:|
| AnnualCrop | 0.87 | 0.92 |
| Forest | 0.88 | 0.99 |
| HerbaceousVegetation | 0.87 | 0.92 |
| Highway | 0.61 | 0.94 |
| Industrial | 0.91 | 0.97 |
| Pasture | 0.82 | 0.95 |
| PermanentCrop | 0.78 | 0.92 |
| Residential | 0.86 | 0.98 |
| River | 0.66 | 0.91 |
| SeaLake | 0.86 | 0.97 |

The CNN's
three most common errors were HerbaceousVegetation to PermanentCrop (15 images), Highway to River (12images), and PermanentCrop to HerbaceousVegetation (10 images). Together they make up roughly a quarter of its errors.
<img width="1790" height="690" alt="image" src="https://github.com/user-attachments/assets/772c3ce7-a0d5-4466-982b-6da550958985" />

## How to run?
Simply find the dataset on Kaggle and open a notebook. Then, run the codes provided in the Jupyter Notebook file in this repository.
