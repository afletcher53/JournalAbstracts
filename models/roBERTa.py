from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split


# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.

dataset_df = pd.read_csv("../data/data_bert.csv")
X = dataset_df.iloc[:,0] 
y = dataset_df.iloc[:,1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

train_df = pd.DataFrame(X_train)
train_df.insert(1, "", y_train)

test_df = train_df = pd.DataFrame(X_test)
test_df.insert(1, "", y_test)

del X_train, X_test, y_train, y_test

model = ClassificationModel('roberta', 'outputs/', args={'fp16': False}) 

# # Train the model
# model.train_model(train_df)

import sklearn


result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)