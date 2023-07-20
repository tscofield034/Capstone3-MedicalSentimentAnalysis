<p align="center">
<br/>
<img src="https://i.imgur.com/JpALNc4.jpg" height="65%" width="65%" alt="brain"/>
<br />

# Medical Text Analysis

*In this work, we used medical abstracts written by doctors to classify whether a patient’s symptoms are due to digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, or general pathological conditions. Furthermore, we compared the results of more traditional methods of preprocessing, extracting features, and classifying NLP data with the results of OpenAI’s chatbot API.*

## 1. Data

The data we are using is a publicly available dataset from Kaggle consisting of examples of medical abstracts for patients with various conditions. The database contains both a training and testing set containing approximately 14000 observations each. However, the testing set does not contain labels, therefore, we will not be using it in this work. Luckily, 14000 observations will be suitable for splitting into testing and training sets.

The text data itself varies from patient to patient ranging from only 24 to 596 words. However, we will investigate whether the length of the abstract, among other characteristics, has an effect on the quality of diagnosis. Also, each observation contains a digit which is the diagnosis label itself. The labels are as follows: 0 = neoplasms, 1 = digestive system diseases, 2 = nervous system diseases, 3 = cardiovascular diseases, and 4 = general pathological conditions.

> * [Medical Text Dataset](https://www.kaggle.com/datasets/chaitanyakck/medical-text?select=train.dat)

## 2. Data Cleaning 

The data needs some initial formatting to be completed to be usable. First, we needed to separate the digits (diagnosis label) from the abstract text itself using regular expressions (RegEx) to find digits followed by a tab. Following this, the observations could be separated into text and label datasets.  

## 3. EDA

Next, in the data science pipeline is exploratory data analysis, or EDA for short. There are not many useful EDA tasks for NLP projects given that all of the data is text-based. However, one thing we can do is investigate our response variable, the diagnosis label, a bit further. In Figure 1, find the distribution of the five response classes. We can see that the most frequent diagnosis is class 4, or general pathological conditions. However, there is not a large class imbalance in the data, so we shouldn’t need to take any special precautions in the modeling stage. The least frequent diagnosis is digestive system diseases. General pathological conditions outweigh digestive system diseases by 4805:1494 or approximately 3.2:1.

<p align="center">
<br/>
<img src="https://i.imgur.com/NUtfe6x.png" height="40%" width="40%" alt="class imbalance"/>
<br />


## 4. Feature Engineering

To understand as much as possible from the diagnosis text, I am going to extract several numerical features from each observation. These numerical features will describe the following: character count, word count, capital word count, quotations count, sentence count unique word count, stopword count, average word length, average sentence length, unique word ratio, stopword ratio, verb count, noun count, adverb count, and adjective count. In total, there are fifteen added features that we can utilize in the modeling stage. Furthermore, we can test our models with and without the added features to determine their value in classification.

## 5. Text Preprocessing

Now, to set the classification algorithms up for the most success, there are several text preprocessing steps to complete. Moreover, I am going to tokenize the text, lowercase all characters, remove the stopwords, lemmatize the text, and execute a TF-IDF vectorizer. In more detail, the tokenizer is used to split up the text string into words, which will be more useful in analysis. Then, it is standard to lowercase all letters so that words can be matched more easily. Stopwords are words that will not be useful for the NLP analysis. Furthermore, they are often “filler” words or other words that allow us to speak fluently and with correct grammar. They are important to everyday life, but the computer will achieve higher classification if these are removed from the data. Next, we lemmatize the text which groups together different inflections of the same word such as “improver”, “improving”, and “improvements” all stemming down to the same word of “improve”. Lastly, we ran the text through a TF-IDF vectorizer which weights the word counts by a measure of how often they appear in the documents. This step should help identify words that are more important for certain medical diagnoses.

**TF-IDF Vectorizer**: Given that we had several thousand unique words in the total dataset, we could not train a TF-IDF vectorizer without assigning it a smaller size as a 14000 x 100000 matrix is not plausible for classification. Therefore, we trained multiple TF-IDF vectorizers using various feature sizes to find the optimal size for our scenario. To test the vectorizers we trained baseline Random Forest and XGBoost classification models and found that a feature size of 2500 is optimal for our situation. A full set of results showing the model’s accuracy and F1-score for various feature sizes can be found in Figure 2. 

<p align="center">
<br/>
<img src="https://i.imgur.com/NUtfe6x.png" height="40%" width="40%" alt="class imbalance"/>
<br />


## 6. Modeling

In the last step of the data science pipeline, I utilized Random Forest, XGBoost, and CatBoost classifiers to measure how accurately medical abstracts could be assigned diagnosis labels. Furthermore, we want to also test the effectiveness of the added numerical features to the medical classification. Therefore, in the first step of modeling, we trained and tested baseline models for all three algorithms using only the text data and using text data plus the numerical features. The full results can be found in Table 1. The best model for both accuracy and F1-score metrics was a CatBoost classifier using the data with the added features. The full classification results can be found in Figure 3. Also, we calculated the feature importance of the added features for this model. A table of the feature importances can be found in Table 2. We can see that the most important added features to the model are unique word ratio, stopword ratio, and noun count.
Next, the logical step of the modeling process is to expand on the baseline models and parameter tune to find an improved model. However, due to the large number of features following Tf-IDF vectorization and our lack of computing power, tuning the CatBoost model was not plausible. Therefore, we parameter-tuned the Random Forest model to determine if the baseline models could be improved. The results of the parameter tuning can be found in Figure 4. As we can see from these results, the accuracy has increased by approximately 4%, but the results themselves are worse if we look at the other metrics and the confusion matrix. Furthermore, the tuned model had many false classifications of class 4 (general pathological conditions). But because this class has the most observations, it benefited the accuracy to predict a majority of the observations in that class. This exercise concluded that the baseline models are satisfactory for this work. Further parameter tuning of the CatBoost classifier could be included in future work to attempt to enhance the model further.

## 5. Future Improvements

* In future work, this model could be refined with additional observations (more patients) and additional features. Perhaps, there are other features that doctors use in practice today to determine if a patient is at risk for strokes that would also be useful for our model. Also, additional parameter tuning could be performed to our existing algorithms. Additonal algorithms such as neural networks could also be applied to the data.
