import pandas as pd

species_train_df = train_df = pd.read_csv('data.csv')
species_train_df = species_train_df.drop(species_train_df.columns[[0,3]], axis = 1)
species_train_df.columns = ['text', 'label']


from simpletransformers.classification import ClassificationModel
import pandas as pd


# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.


# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-cased', num_labels=2, args={'reprocess_input_data': True,  'fp16': False,'overwrite_output_dir': True}) 
# You can set class weights by using the optional weight argument

# Train the model
model.train_model(species_train_df)

