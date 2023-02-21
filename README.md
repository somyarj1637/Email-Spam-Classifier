
# Email Spam Classifier

This project is a machine learning model that can classify emails as spam or not spam. The model is trained on a dataset of labeled emails and uses a combination of text preprocessing and natural language processing techniques to extract features from the email text.

This email spam classifier is designed to identify whether an email is spam or not. The model uses natural language processing (NLP) techniques to analyze the content of an email and predict whether it is likely to be spam or not.


## Dataset Used:
The dataset used to train the model is the Spam Assassin Public Corpus. It consists of over 5,000 labeled emails, with approximately 75% of the emails labeled as spam.

#### Link to the Dataset: 


## Preprocessing :
Before the emails are fed into the model, they undergo a preprocessing step. This includes removing stop words, stemming, and converting all text to lowercase. Additionally, the email text is tokenized, and each token is converted to a numerical feature using the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm.

Formally the steps followed are as follows:
* Lower case
* Tokenization
* Removing Special Character
* Removing stop words
* Stemming
## Model Building

The model used in this project is a Naive Baye's classifier. The model was chosen due to its high accuracy and ability to handle high-dimensional feature spaces. The model was trained using vectorization to prevent overfitting.
## Result

The model achieves an accuracy of approximately 98% on the test dataset. This indicates that the model is very good at distinguishing between spam and non-spam emails.
## Future Work

Possible future work includes expanding the dataset to include more recent emails, incorporating additional features such as email metadata, and exploring other machine learning algorithms to compare performance.
## Usage

To use the model, simply provide an email as input and the model will output a binary classification indicating whether the email is spam or not. This can be done using a simple Python script or through a web-based interface.
## Requirements :

The following packages are required to run the email spam classifier:

* Python (version 3.6 or higher)
* Scikit-learn
* Pandas
* Numpy
* Streamlit
## Installation

To install the required packages, run the following command:

#### Code: `pip install -r requirements.txt`
 
## How to use: 

To use the email spam classifier, follow these steps:

* Open app.py in your text editor of choice.
* Edit the email_text variable to include the text of the email you want to classify.
* Run the app.py file using the following command:

#### Code: `python app.py`
## Open for Collaboration

* Feel free to collaborate 
* Star this repo
* Fork it and use it
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
