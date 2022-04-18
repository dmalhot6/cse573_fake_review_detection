import os
import pickle
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

categories = ["Books_5", "Clothing_Shoes_and_Jewelry_5", "Electronics_5", "Home_and_Kitchen_5", "Kindle_Store_5",
              "Movies_and_TV_5", "Pet_Supplies_5", "Sports_and_Outdoors_5", "Tools_and_Home_Improvement_5",
              "Toys_and_Games_5"]

categories_str = "Clothing_Shoes_and_Jewelry_5"
for i in range(1, len(categories)):
    categories_str += ", " + categories[i]


def countvectorize(statement):
    vectorizer = pickle.load(open(os.path.join("models", "vectorizer.pk"), 'rb'))
    statement = vectorizer.transform(statement).toarray()
    return statement


def onehotencode(rating, product_category, X):
    le_1 = pickle.load(open(os.path.join("models", "le_1.pk"), 'rb'))
    le_3 = pickle.load(open(os.path.join("models", "le_3.pk"), 'rb'))

    col_trans_a = pickle.load(open(os.path.join("models", "col_trans_a.pk"), 'rb'))
    col_trans_c = pickle.load(open(os.path.join("models", "col_trans_c.pk"), 'rb'))

    w = 2
    h = 1
    column_label_encoding = [[0 for x in range(w)] for y in range(h)]

    for i in range(0, 1):
        column_label_encoding[i][0] = rating

        column_label_encoding[i][1] = product_category

    column_label_encoding = np.array(column_label_encoding)

    column_label_encoding[:, 0] = le_3.transform(column_label_encoding[:, 0])
    column_label_encoding[:, 1] = le_1.transform(column_label_encoding[:, 1])

    column_label_encoding = col_trans_a.transform(column_label_encoding)
    try:
        column_label_encoding = column_label_encoding.toarray()
    except:
        pass
    column_label_encoding = column_label_encoding.astype(np.float64)


    try:
        column_label_encoding = column_label_encoding.toarray()
    except:

        pass
    column_label_encoding = column_label_encoding.astype(np.float64)

    column_label_encoding = col_trans_c.transform(column_label_encoding)
    try:
        column_label_encoding = column_label_encoding.toarray()
    except:
        pass
    column_label_encoding = column_label_encoding.astype(np.float64)

    X = np.append(X, column_label_encoding, axis=1)
    print(len(X))
    return X


def get_POS_Tagging(sentence_to_tag):
    total_verb_count = 0
    total_noun_count = 0

    pos_tags = []

    text = nltk.word_tokenize(sentence_to_tag)
    tagged_words_list = (nltk.pos_tag(text))

    for lis in tagged_words_list:
      pos_tags.append(lis[1])

    for tag in pos_tags:
        if tag in ['NOUN','NNP','NN','NUM','NNS','NP','NNPS']:
            total_noun_count += 1
        elif tag in ['VERB','VB','VBN','VBD','VBZ','VBG','VBP']:
            total_verb_count += 1
        else:
            continue

    if total_noun_count >= total_verb_count:
        sentence_to_tag = 'T'
    else:
        sentence_to_tag = 'F'

    return sentence_to_tag


def postag(sentence, X):
    width = 2
    height = 1

    text_pos_tag = []
    for i in range(height):
      row_list = []
      for j in range(width):
        row_list.append(0)
      text_pos_tag.append(row_list)

    pos_tagged_sentence = get_POS_Tagging(sentence)

    if pos_tagged_sentence == 'T':
        text_pos_tag[0][0] = 1
        text_pos_tag[0][1] = 0
    else:
        text_pos_tag[0][0] = 0
        text_pos_tag[0][1] = 1

    X = np.append(X, text_pos_tag, axis=1)
    return X


def classify(X):
    rfc = pickle.load(open(os.path.join("models", "randomforest.pk"), 'rb'))
    return rfc.predict(X)


def get_result(statement, rating, product_category):
    X = countvectorize([statement])
    X = postag(statement, X)
    X = onehotencode(rating, product_category, X)

    X = classify(X)
    return X


def test_input(product_rating, product_category):
    x = True
    y = True
    z = True

    if product_rating != '1' and product_rating != '2' and product_rating != '3' and product_rating != '4' and product_rating != '5':
        print()
        print()
        print("\nError : Rating of a product must be from 1 to 5.")
        print("\nPlease try again")

        x = False

    if product_category not in categories:
        print()
        print()
        print("\nError : Categories must be one of the following : \n" + categories_str)
        print("\nPlease try again")

        z = False

    return [x, y, z]


if __name__ == '__main__':

    review_text = input("\nPlease type in your review : ")

    product_rating = ""

    product_category = ""

    input_ar = [False, False, False]

    while (True):
        if not input_ar[0]:
            product_rating = input("\nPlease enter your rating from 1 to 5 for the product : ")
            product_rating = '3'

        if not input_ar[2]:
            product_category = 'Home_and_Kitchen_5'
        input_ar = test_input(product_rating, product_category)

        if input_ar == [True, True, True]:
            break

    answer = get_result(review_text, product_rating, product_category)

    if answer == 1:
        print("This is an Actual review")

    else:
        print("This is a Fake review")


