# Test model
# Test model
from pprint import pprint
from src.model import Model
from src.interface import Interface
from src.utils.loader import load_data
from src.utils.vocab import Vocab, Indexer
import os
import math
import random
import torch
from tqdm import tqdm


class Testor:
    def __init__(self, model_path, data):
        '''
            example:
            model_path='./models/snli/benchmark-0/best.pt'
            data=['i am tom', 'my name is tom']
        '''
        self.model, self.checkpoint= Model.load(model_path)
        self.args = self.checkpoint['args']
        self.data_file='./data/snli/pri.txt'
        with open(self.data_file, 'w+') as f:
            f.write(data[0]+"\t"+data[1]+"\t"+'0')
        self.data=load_data(*os.path.split(self.data_file))
        
        self.inputs=self.process_data()


    def process_data(self):
        
        interface = Interface(self.args)
        inputs = interface.pre_process(self.data, training=False)
        return inputs  

    def Run(self):
        
        pred, prob = self.model.test(self.inputs)
        print(prob)
        return [{
            'Độ tương đồng': "{:.4}%".format(prob[0][0]*100),
        }]
        