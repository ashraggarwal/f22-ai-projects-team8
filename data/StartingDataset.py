import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """
    def __init__(self, data_path,train=True):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # Preprocess the data. These are just library function calls so it's here for you
        self.df = pd.read_csv(data_path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        self.labels = self.df.target.tolist() # list of labels
        if(train==True):
            self.labels = self.labels[0:int(0.9*len(self.df))]
            self.sequences = self.sequences[0:int(0.9*len(self.df)),:]
        else:
            self.labels = self.labels[int(0.9*len(self.df))+1:]
            self.sequences = self.sequences[int(0.9*len(self.df))+1:,:]
        print(self.sequences.shape[0],len(self.labels))
        self.token2idx = self.vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards
        #print(self.sequences[0]) #if you don't use toarray it'll look weird probably because the object returned by fit_transform doesn't have well defined behavior for print
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label
        return self.sequences[i, :].toarray(), self.labels[i]

    def __len__(self):
        return self.sequences.shape[0]