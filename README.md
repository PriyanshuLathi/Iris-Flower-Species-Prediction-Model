# Iris Flower Species Prediction Model

This project focuses on building a Support Vector Machine (SVM) model to classify the Iris dataset. The Iris dataset is a classic dataset in machine learning that contains data on three different species of iris flowers: Setosa, Versicolor, and Virginica. The goal is to predict the species of an iris flower based on the features provided.

## Dataset Description

The dataset used in this project was obtained from Scikit-learn's built-in datasets module. It is a well-known dataset frequently used for benchmarking classification algorithms.

The dataset consists of 150 samples, with each sample containing four features:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

Each sample is labeled with one of three target classes representing the species of the iris flower.

<p align="center"> <img src="https://github.com/user-attachments/assets/08294565-6d81-4e32-bf32-2f8cfc3194d6" alt="USA house price prediction" width = 1000 /> </p>

## Methodology

The following steps were taken to develop the SVM model:

1. **Data Loading and Exploration**: The Iris dataset was loaded using Scikit-learn's built-in dataset library. Basic data exploration was conducted to understand the feature distributions and target labels.

2. **Data Preprocessing**: 
   - The dataset was split into training (70%) and testing (30%) sets.
   - No additional feature scaling was applied since SVM models are generally robust to feature scaling due to kernel functions.

3. **Model Training**: 
   - An SVM model was trained using the default hyperparameters.
   - The model was fit using the training data, learning to classify the iris species based on the features.

4. **Model Evaluation**: 
   - Predictions were made on the test set.
   - The model was evaluated using a confusion matrix and a classification report, which includes precision, recall, and F1-score for each class.

## Accuracy and Model Performance

**Accuracy**: The model achieved an overall accuracy of **98%**, correctly classifying 98% of the test samples.

<p align="center"> <img src="https://github.com/user-attachments/assets/6682e1cd-a50c-422c-a447-cc05ece5d0f0" alt="USA house price prediction" width = 1000 /> </p>

## Future Scope

There are several avenues for enhancing and expanding this project:

- **Hyperparameter Tuning**: The SVM model could benefit from fine-tuning hyperparameters such as the kernel type, C value, and gamma to potentially improve classification performance.
- **Feature Scaling**: Although SVMs are generally robust, scaling features using techniques like StandardScaler or MinMaxScaler could further optimize model performance.
- **Comparison with Other Models**: Comparing the SVM model's performance against other classification algorithms like Decision Trees, Random Forests, or Neural Networks would provide insights into the strengths and weaknesses of different approaches.
- **Data Augmentation**: Techniques like synthetic data generation or bootstrapping could be used to enhance the training data, potentially improving the model’s generalizability.
- **Real-Time Deployment**: Implementing the model as part of an interactive web application or API would showcase its practical use in classifying iris species from new input data.

## Dependencies

To run this project, you'll need the following libraries:

- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone 
    ```

2. Install the required Python dependencies:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/PriyanshuLathi/Iris-Flower-Species-Prediction-Model/blob/main/LICENSE) file for details.

## Contact

For questions or feedback, feel free to reach out:

- LinkedIn: [Priyanshu Lathi](https://www.linkedin.com/in/priyanshu-lathi)
- GitHub: [Priyanshu Lathi](https://github.com/PriyanshuLathi)

## Authors
- Priyanshu Lathi