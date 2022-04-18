import os
import numpy as np
import pickle
import nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

#############################################################################################################
# Tokenization and Stemming
#############################################################################################################
from tokenization import perform_tokenization

reviews, amazon_dataset = perform_tokenization()
length = len(amazon_dataset)
########## COUNT VECTORIZATION ############################


vectorizer=CountVectorizer()
X=vectorizer.fit_transform(reviews).toarray()
y=amazon_dataset.iloc[:length,2]

filename = 'vectorizer.pk'
pickle.dump(vectorizer, open(os.path.join("models", filename), 'wb'))


############ POS TAGGING #####################################
from pos_tagging import get_POS_Tagging
print("\n Started with POS Tagging \n")
width = 2
height = length
total_iteration = 0

text_pos_tag = []
for i in range(height):
    row_list = []
    for j in range(width):
        row_list.append(0)
    text_pos_tag.append(row_list)

load_flag = False
pos_tag_file = 'pos_tag3.pk'
print("\n\nExecuting POS tagging for each sentence")

for i in range(0, length):
    if os.path.exists(os.path.join("models", pos_tag_file)):
        load_from_disk = True
        continue
    sentence_to_tag = amazon_dataset['text_'][i]
    pos_tagged_sentence = get_POS_Tagging(sentence_to_tag)

    if pos_tagged_sentence != 'T':
        text_pos_tag[i][0] = 0
        text_pos_tag[i][1] = 1
    else:
        text_pos_tag[i][0] = 1
        text_pos_tag[i][1] = 0

if load_flag != True:
    pickle.dump(text_pos_tag, open(os.path.join("models", pos_tag_file), 'wb'))

if load_flag == True:
    text_pos_tag = pickle.load(open(os.path.join("models", pos_tag_file), 'rb'))

X = np.append(X, text_pos_tag, axis=1)
print("\n POS Tagging successful \n")

############ LABEL ENCODING ###############################
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

width = 2
height = length

column_label_encoding = []
for i in range(height):
  row_list = []
  for j in range(width):
    row_list.append(0)
  column_label_encoding.append(row_list)

test = dict()


for i in range(0, length):
    column_label_encoding[i][1] = amazon_dataset["category"][i]
    column_label_encoding[i][0] = amazon_dataset["rating"][i]

    if column_label_encoding[i][1] not in test.keys() :
        test[column_label_encoding[i][1]] = 1

column_label_encoding = np.array(column_label_encoding)

column_label_encoding[:, 1] = label_encoder.fit_transform(column_label_encoding[:, 1])
filename = 'le_1.pk'
pickle.dump(label_encoder, open(os.path.join("models", filename), 'wb'))

column_label_encoding[:, 0] = label_encoder.fit_transform(column_label_encoding[:, 0])
filename = 'le_3.pk'
pickle.dump(label_encoder, open(os.path.join("models", filename), 'wb'))

############ ONEHOT ENCODER #################################
col_trans_a = ColumnTransformer([("rating", OneHotEncoder(), [0])], remainder ='passthrough')
column_label_encoding = col_trans_a.fit_transform(column_label_encoding).astype(np.float32)

filename = 'col_trans_a.pk'
pickle.dump(col_trans_a, open(os.path.join("models", filename), 'wb'))
col_trans_c = ColumnTransformer([("category", OneHotEncoder(), [5])], remainder ='passthrough')
column_label_encoding = col_trans_c.fit_transform(column_label_encoding).astype(np.float32)
filename = 'col_trans_c.pk'
pickle.dump(col_trans_c, open(os.path.join("models", filename), 'wb'))
column_label_encoding = column_label_encoding.astype(np.uint8)
X = X.astype(np.uint8)
X = np.append(X, column_label_encoding, axis=1).astype(np.uint8)

############ TEST AND TRAIN ###################################
training_X, testing_X, training_Y, testing_Y = train_test_split(X, y, test_size = 0.2, random_state = 1)

#### TRAIN ##########
############### Random Forest Classifier ############
print("\n Training Random forest and perfoming analysis \n")
rfc = None
if os.path.exists(os.path.join("models", "randomforest.pk")):
    rfc = pickle.load(open(os.path.join("models", "randomforest.pk"), "rb"))
else:
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(training_X, training_Y)

filename = 'randomforest.pk'
pickle.dump(rfc, open(os.path.join("models", filename), 'wb'))
y_pred_rfc = rfc.predict(testing_X)

print ("\nAccuracy of Random forest is : ")
acc_rf = accuracy_score(testing_Y, y_pred_rfc) * 100
print (acc_rf)

print ("\nPrecision of Random forest is : ")
prec_rf = precision_score(testing_Y, y_pred_rfc) * 100
print (prec_rf)

print ("\nf1 score of Random forest is : ")
f1_rf = f1_score(testing_Y, y_pred_rfc, average='macro') * 100
print (f1_rf)

########### KNN classifier #######################
print("\n Training KNN(K-7) and perfoming analysis \n")
neigh = None
filename = 'knn.pk'
if os.path.exists(os.path.join("models", "knn.pk")):
    print("we are in if cond")
    neigh = pickle.load(open(os.path.join("models", "knn.pk"), "rb"))
else:
    print("we are in else cond")
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(training_X, training_Y)



pickle.dump(neigh, open(os.path.join("models", filename), 'wb'))

y_pred_neigh = neigh.predict(testing_X)

print ("\nAccuracy of KNN is : ")
acc_knn = accuracy_score(testing_Y, y_pred_neigh) * 100
print (acc_knn)


print ("\nPrecision of KNN is : ")
prec_knn = precision_score(testing_Y, y_pred_neigh) * 100
print (prec_knn)

print ("\nf1 score of KNN is : ")
f1_knn = f1_score(testing_Y, y_pred_neigh, average='macro') * 100
print (f1_knn)

plt.legend()
knn_metric = [acc_knn, f1_knn, prec_knn]
rfc_metric = [acc_rf, f1_rf, [prec_rf]]
x_axis_label = ['Accuracy','f1 score', 'Precision']
X_axis = np.arange(len(x_axis_label))
plt.bar(X_axis - 0.2, knn_metric, 0.4, label = 'KNN')
plt.bar(X_axis + 0.2, rfc_metric, 0.4, label = 'Random Forest')

plt.xticks(X_axis, x_axis_label)
plt.xlabel("Metric")
plt.ylabel("Value")
plt.title("comparison")
plt.legend()
plt.show()

######### Fin ###################
