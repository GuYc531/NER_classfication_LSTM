### Guy Cohen
This is my explanation and understanding of the exersice:

I chose to load and process the dataset into pandas DataFrame because of convenience reasons only,
in production env I would consider to preprocess across the dataset object to save memory and computation time.

I will describe the pipeline steps:

## DataHandler
class which handle, preprocess the dataset

### create_data_frames()

First, I created a DataFrame for each part of the dataset train,validation and test,
with columns 'tags' and 'tokens'.

1. when loading the data, I joined the tokens into 1 string
2. I used padding for the tags according to the max length token length in the dataset

### tokenize_sentences_in_data_frames()

I created a new column 'tokenized_tokens_column' as its represents the tokenizer output of the 'tokens' column

1. creating a new column of the tokenizer embedding output for the 'token' column
2. padding the tokenizer output for the 'max_length_sequence' to keep the dimensions equal for all samples
3. validate the samples dimensions in all dataframes in function _validate_tokens_length()

### create_data_loaders()

I created a custom dataloader object for each dataset part train, validation and test

1. I used the NERDataset class which acts as my custom dataset object (inheres from Dataset class)
2. Constructed a DataLoader from the custom dataset and saves it as a class attribute

## Training_lstm

class which includes the training stage of a given model

### train()

function which train the give model, the model i used is in class LSTMNERModel.
A basic LSTM model from pytorch.

the model has a fully connected output layer which used to predict the most probable class of NER according to the model. 

1. given the hyperparameters from main.py the model will train on the DataLoaders in the data_handler object.
2. for every epoch it will train on the training set, and after will test on the validation set.
3. at every epoch i will save the best model in a file in directory 'models/'
4. I will save the losses and validation accuracy for further analysis if needed

### plot_results()

A function that utilize util function plot_losses which given the train and val loss and val accuracy,
provides plot of their progress across epochs and saves it locally


### test_over_test_data_set()

a function that test the model over the test part of the dataset,
and plots a classification report and a confusion matrix if needed.


## general Notes:

1. when predicting validation and test i used torch.no_grad() in order for the model not to train.
2. I chose the pretrained tokenizer "bert-base-uncased" as it will probably generalize well over the data.
3. I did not configure the parameters of the LSTM model too much because I understood this is not the main focus of this exercise.
4. I tried to build and write the code "production ready" the most i could due to the time limit
5. I handled the padding with 'padded_index' parameter so the model will not learn or train over this labels
6. the best test i could think of to evaluate the model is the classification report, 
which in it i can see the metrics for each class of 'labels_list' and to get a better understand what my model knows better and the opposite.