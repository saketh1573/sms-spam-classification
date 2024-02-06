# SMS Spam Classification

This repository contains code for building a spam classification model using the SMS Spam Collection dataset. The goal is to classify SMS messages as either spam or ham (non-spam) using machine learning algorithms.

## Dataset

The dataset used in this project is the SMS Spam Collection, which consists of SMS messages labeled as spam or ham. The dataset is provided in the file `SMSSpamCollection`, and it is loaded into a pandas DataFrame for further processing.

## Libraries Used

The following libraries were used in this project:
- `numpy` and `pandas` for data manipulation and analysis
- `matplotlib` and `seaborn` for data visualization
- `nltk` for natural language processing tasks such as text cleaning and lemmatization
- `sklearn` for machine learning algorithms and evaluation metrics

## Project Steps

1. Data Exploration:
   - Descriptive statistics of the dataset
   - Visualization of the class distribution using a count plot

2. Data Preprocessing:
   - Mapping the label values to numerical values (0 for ham, 1 for spam)
   - Balancing the dataset by oversampling the minority class (spam)
   - Creating additional features based on currency symbols and numbers present in the messages

3. Text Preprocessing:
   - Removing special characters and numbers from the messages
   - Converting messages to lowercase
   - Removing stopwords (common words with little significance)
   - Lemmatizing words (reducing words to their base or root form)

4. Feature Extraction:
   - Using TF-IDF vectorization to convert text data into numerical feature vectors

5. Model Building and Evaluation:
   - Splitting the dataset into training and testing sets
   - Training a Multinomial Naive Bayes model and evaluating its performance using cross-validation and classification report
   - Training a Decision Tree model and evaluating its performance using cross-validation and classification report

## Models Used

1. Multinomial Naive Bayes
   - Naive Bayes is a probabilistic classification algorithm that is commonly used for text classification tasks.
   - The Multinomial Naive Bayes variant is suitable for discrete features such as word counts.
   - It assumes that features are conditionally independent given the class.
   - The model calculates the probability of a message belonging to each class (spam or ham) based on the occurrence of words in the message.
   - The model is trained using the training set and evaluated using cross-validation and classification report.

2. Decision Tree
   - Decision Tree is a non-parametric supervised learning algorithm that can be used for both classification and regression tasks.
   - It builds a tree-like model of decisions and their possible consequences.
   - The tree is constructed by recursively splitting the dataset based on features to maximize information gain or Gini impurity.
   - Each internal node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome (class label).
   - The model is trained using the training set and evaluated using cross-validation and classification report.

## Technical Information

The code is written in Python and utilizes Jupyter Notebook for an interactive and reproducible workflow. Here are some technical details about the implementation:

- The code is implemented using the Python programming language (Python 3.7+).
- The machine learning models are built using the scikit-learn library (`sklearn`).
- Natural language processing tasks such as text cleaning and lemmatization are performed using the Natural Language Toolkit (`nltk`).
- Text data is converted into numerical feature vectors using the TF-IDF vectorization technique

## Usage

You can use this code to build your own spam classification model using the SMS Spam Collection dataset. Follow the steps outlined in the Jupyter Notebook to preprocess the data, extract features, and train different machine learning models. Feel free to modify the code as needed to suit your requirements.

## Conclusion

By building and evaluating different machine learning models, you can choose the best model for classifying SMS messages as spam or ham. The results and insights gained from this project can be used for various applications such as email filtering, message prioritization, and detecting fraudulent messages.

## Credits

- The SMS Spam Collection dataset was obtained from [Kaggle](https://www.kaggle.com/datasets).
- The code in this repository was written by Sai Saketh Motamarry.
