import codecs
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import os
import pandas as pd

def perform_tokenization():
    with codecs.open("fake_reviews_dataset.csv", "r", encoding='utf-8', errors='ignore') as data_file:
        amazon_dataset = pd.read_csv(data_file)


    length_dataset = len(amazon_dataset)
    print("\nDataset read successful\n")
    print("\nStarted performing Tokenization and Stemming.")
    reviews = []
    stopwords_set = set(stopwords.words('english'))

    filename = 'reviews.pk'

    if os.path.exists(os.path.join("models", "reviews.pk")):
        reviews = pickle.load(open(os.path.join("models", filename), 'rb'))

    else:
        for i in range(length_dataset):
            review = re.sub('[^a-zA-Z]', ' ', amazon_dataset['text_'][i])
            review = review.lower().split()
            review = [word for word in review if word not in stopwords_set]
            stemmer = PorterStemmer()
            review = ' '.join([stemmer.stem(word) for word in review])
            reviews.append(review)

        pickle.dump(reviews, open(os.path.join("models", filename), 'wb'))

    print("\n Tokenization and Stemming completed\n")
    return reviews, amazon_dataset