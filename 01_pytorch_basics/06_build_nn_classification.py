#%%
import torch
torch.__version__
import pandas
from torch.utils.data import Dataset

#%%
class OneHotEncoder():
    def __init__(self, series):
        '''Given a single pandas series, creaet an encoder
        that can turn values from that series into a one hot
        pytorch tensor.
        
        Arguments:
            series {pandas.Series} -- encode this
        '''
        unique_values = series.unique()
        self.ordinals = {
            val: i for i, val in enumerate(unique_values)
            }
        self.encoder = torch.eye(
            len(unique_values), len(unique_values)
            )

    def __getitem__(self, value):
        '''Turn a value into a tensor
        
        Arguments:
            value {} -- Value to encode, 
            anything that can be hashed but most likely a string
        
        Returns:
            [torch.Tensor] -- a one dimensional tensor
        '''

        return self.encoder[self.ordinals[value]]
    
#%%
class MushroomsCategoricalCsv(Dataset):
    def __init__(self, datafile, output_column_name) -> None:
        '''Load the dataset and create needed encoders for
        each series.
        
        Arguments:
            datafile {string} -- path to data file
            output_column_name {string} -- series/column name
        '''
        super().__init__()
        self.data = pandas.read_csv(datafile)
        self.output_column_name = output_column_name
        self.encoders = {}
        for series_name, series in self.data.items():
            self.encoders[series_name] = OneHotEncoder(series)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''Return an (input, output) tensor tuple
        with all categories one hot encoded.
        
        Arguments:
            index {[type]} -- [description]
        '''
        if type(idx) is torch.Tensor:
            idex = idx.item()
        sample_data = self.data.iloc[idx]
        output = self.encoders[self.output_column_name][sample_data[self.output_column_name]]
        input_components = []
        for name, value in sample_data.items():
            if name != self.output_column_name:
                input_components.append(self.encoders[name][value])
        input = torch.cat(input_components)
        return input, output
#%%
class MushroomClassificationModel(torch.nn.Module):
    def __init__(self, input_dimensions,output_dimensions, size=128) -> None:
        super().__init__()
        self.layer_one = torch.nn.Linear(input_dimensions, size)
        self.layer_one_activation = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.layer_two_activation = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, output_dimensions)
    
    def forward(self, inputs):
        buffer = self.layer_one(inputs)
        buffer = self.layer_one_activation(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.layer_two_activation(buffer)
        buffer = self.shape_outputs(buffer)
        return torch.nn.functional.softmax(buffer, dim=-1)
                
#%%
# Mushroom dataset from Kaggle
class MushroomDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pandas.read_csv('./mushrooms.csv')

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int, tensor} -- data at this position.
        '''
        # handle being passed a tensor as an index
        if type(idx) is torch.Tensor:
            idx = idx.item()
        # Below self.data.iloc[idx][1:] -> are features (x0, x1, x2 etc)
        # Below self.data.iloc[idx][0:1] -> are outputs (y values)     
        return self.data.iloc[idx][1:], self.data.iloc[idx][0:1]
    
mushrooms = MushroomDataset()
print(len(mushrooms))
print(mushrooms[0])

# 5% data for testing
no_of_test_rows = int(len(mushrooms) * 0.05)
no_of_train_rows = len(mushrooms) - no_of_test_rows

train_data, test_data = torch.utils.data.random_split(mushrooms, (no_of_train_rows,no_of_test_rows))
print(len(train_data))
print(len(test_data))


mushrooms = MushroomsCategoricalCsv('./mushrooms.csv', 'class')
mushrooms[0]

model = MushroomClassificationModel(mushrooms[0][0].shape[0], mushrooms[0][1].shape[0])
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCELoss()

number_for_testing = int(len(mushrooms) * 0.05)
number_for_training = len(mushrooms) - number_for_testing
train, test = torch.utils.data.random_split(mushrooms,
    [number_for_training, number_for_testing])
training = torch.utils.data.DataLoader(train, 
    batch_size=16, shuffle=True)
for epoch in range(4):
    for inputs, outputs in training:
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(loss))

import sklearn.metrics

testing = torch.utils.data.DataLoader(test, 
    batch_size=len(test), shuffle=False)
for inputs, outputs in testing:
    results = model(inputs).argmax(dim=1).numpy()
    actual = outputs.argmax(dim=1).numpy()
    accuracy = sklearn.metrics.accuracy_score(actual, results)
    print(accuracy)
    
# and, you can see how accurate you are -- per class this is
# a way to tell if you model is better or worse at making i
# certain
# kinds of predictions

#%%
sklearn.metrics.confusion_matrix(actual, results)

#%%
# you read this left to right
# true positive, false positive
# false negative, true negative

#%%
# even better, you can get a handy classification report, 
# which is easy to read

#%%
print(sklearn.metrics.classification_report(actual, results))