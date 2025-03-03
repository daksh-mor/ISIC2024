# ISIC 2024 Skin Cancer Detection

This repository contains the code for a skin cancer detection system developed for the ISIC 2024 Challenge. The system utilizes deep learning and machine learning techniques to classify skin lesions from 3D total body photos, aiming to improve early skin cancer detection.

## Project Overview

![Uploading image.png…]()


This project focuses on developing a robust and accurate binary classification model to identify histologically confirmed skin cancer cases from single-lesion crops of 3D total body photos. The image quality resembles close-up smartphone photos, making the model applicable in telehealth settings where access to specialized care is limited.

Inspired by successful approaches in similar competitions, this project explores a multi-modal ensemble strategy and employs advanced techniques to handle class imbalance and optimize model performance.

## Features

* **Deep Learning Model:** Utilizes a pre-trained DINOv2 model fine-tuned for skin lesion classification.
* **Data Preprocessing:** Implements image resizing, normalization, and tensor conversion using `albumentations`.
* **Class Imbalance Handling:** Employs undersampling to balance the training dataset.
* **Training and Evaluation:** Implements a training loop with Binary Cross Entropy Loss and NAdam optimizer, evaluating performance using partial AUC (pAUC).
* **HDF5 Data Loading:** Efficiently loads image data from HDF5 files.
* **Transformers usage:** Utilizes pretrained transformers models.
* **Scoring function:** Correctly implements the competition scoring function.

## Technologies Used

* Python
* PyTorch
* Albumentations
* Transformers
* Pandas
* NumPy
* H5py

## Getting Started

### Prerequisites

* Python 3.x
* PyTorch
* Transformers
* Albumentations
* Pandas
* NumPy
* H5py
* torchinfo (optional, for model summary)

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/ISIC2024.git](https://www.google.com/search?q=https://github.com/your-username/ISIC2024.git)
    cd ISIC2024
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download the ISIC 2024 dataset and place the `train-metadata.csv` and `train-image.hdf5` files in the appropriate directory.
4. Download the pretrained DINOv2 model and processor and place them in the correct directory.

### Usage

1.  Run the Jupyter notebook `dinov2.ipynb` to train and evaluate the model:

    ```bash
    jupyter notebook dinov2.ipynb
    ```

2.  Follow the instructions in the notebook to execute the code.

## Future Improvements

* **Fine-tuning DINOv2:** Unfreeze and fine-tune layers of the pre-trained DINOv2 model for better performance.
* **Advanced Data Augmentation:** Implement more extensive data augmentation techniques (e.g., rotations, flips, color jitter).
* **Class Imbalance Techniques:** Explore oversampling, weighted loss functions, or other methods to handle class imbalance more effectively.
* **Hyperparameter Tuning:** Conduct thorough hyperparameter tuning to optimize model performance.
* **Ensemble Methods:** Implement ensemble methods to combine multiple models and improve accuracy.
* **Early Stopping:** Add early stopping to the training loop to prevent overfitting.
* **Model Checkpointing:** Implement model checkpointing to save the best performing models.
* **CV Strategy:** Consider Triple K-Fold CV strategy for more robust testing.
* **LightGBM integration:** Explore integrating LightGBM with the deep learning model as an ensemble.
* **Optuna integration:** Implement Optuna for hyperparameter optimization.

## Reference Project Experience

**Skin Cancer Detection System**

* Developed a multi-modal ensemble by stacking CNN outputs (EfficientNet, ResNet, DenseNet) with LightGBM for the ISIC 2024 Challenge, improving classification performance on imbalanced lesion datasets.
* Engineered a robust Triple K-Fold CV strategy to handle severe class imbalance, ensuring stable and reliable AUC-ROC evaluation across folds.
* Conducted 75+ Kaggle submissions, iterating over architecture refinements and hyperparameter tuning via Optuna, leading to an optimized LightGBM + EfficientNet model with superior generalization.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to submit a pull request.

## License

This project is licensed under the MIT License.
