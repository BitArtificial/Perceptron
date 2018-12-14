# -*- coding: utf-8 -*-
import os
import csv
from perceptron import Perceptron
import matplotlib.pyplot as plt

data_path=r'./Caesarian.txt'



def load_data(data_path):
    X=[]
    Y=[]
    with open(data_path,'r') as f:
        content=csv.reader(f)
    #    print(len(content))
        for i,line in enumerate(content):
            num_of_line=len(line)
            row=[]
            for j,ele in enumerate(line):
                if j==num_of_line-1:
                    y=-1 if eval(ele)==0 else 1
                    Y.append(y)
                else:
                    row.append(eval(ele))
            X.append(row)
    return X,Y

  
X,Y=load_data(data_path)
n=len(X)
d=len(X[0])

per=Perceptron(d,n,X,Y)
per.gradiant_method()

