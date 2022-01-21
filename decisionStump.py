# -*- coding: utf-8 -*-
"""
10601 - Intro to Machine Learning
Homework 1

@author: Hoi Kit Fu
"""

import numpy as np
import csv
import sys

        

if __name__ == '__main__':
    infile_train = sys.argv[1]
    infile_test = sys.argv[2]
    feature_index = sys.argv[3]
    outfile_train = sys.argv[4]
    outfile_test = sys.argv[5]
    metrics = sys.argv[6]
    tsv_data = []

    
    feature_index = int(feature_index)

    with open(infile_train, 'r') as tsv_in_train:
        tsv_reader = csv.reader(tsv_in_train, delimiter='\t')
        tsv_labels = tsv_reader.__next__()
   
        for record in tsv_reader:
            tsv_data.append(record)
        tsv_in_train.close()
        
    np_tsv_data = np.array(tsv_data)
    
    #divide the dataset on that attribute 
    np_tsv_data_g1 = []
    np_tsv_data_g2 = []
        
    for i in range(len(np_tsv_data)):
        if np_tsv_data[i][feature_index] == np.unique(np_tsv_data[:, feature_index])[0]:
            np_tsv_data_g1.append(np.array(np_tsv_data[i]))
        else:
            np_tsv_data_g2.append(np.array(np_tsv_data[i]))            


    np_tsv_data_g1 = np.stack(np_tsv_data_g1, axis=0)
    np_tsv_data_g2 = np.stack(np_tsv_data_g2, axis=0)
    
    output_col = np_tsv_data.shape[1] - 1
 
        
    #make two majority vote decision for each dataset for that attribute
    if (np.sum(np_tsv_data_g1[:, output_col] == np.unique(np_tsv_data[:, output_col])[0]) > np.sum(np_tsv_data_g1[:, output_col] == np.unique(np_tsv_data[:, output_col])[1])):
        vote_group_1 = np.unique(np_tsv_data[:,output_col])[0]
    else:
        vote_group_1 = np.unique(np_tsv_data[:,output_col])[1]
    
    
    if (np.sum(np_tsv_data_g2[:, output_col] == np.unique(np_tsv_data[:, output_col])[0]) > np.sum(np_tsv_data_g2[:, output_col] == np.unique(np_tsv_data[:, output_col])[1])):
        vote_group_2 = np.unique(np_tsv_data[:,output_col])[0]
    else:
        vote_group_2 = np.unique(np_tsv_data[:,output_col])[1]
    
    #prediction for training dataset
    predicted_label_train = []
    
    for i in range(len(np_tsv_data)):
        if np_tsv_data[i][feature_index] == np.unique(np_tsv_data[:, feature_index])[0]:
            predicted_label_train.append(vote_group_1)
        else:
            predicted_label_train.append(vote_group_2)
    
    #metrics for training dataset
    error_train = 1 - (np.sum(np_tsv_data[:, output_col] == predicted_label_train) / np_tsv_data.shape[0])
        

    
    #read the test data
    tsv_data_test = []
    with open(infile_test, 'r') as tsv_in_test:
        tsv_reader = csv.reader(tsv_in_test, delimiter='\t')
        

        tsv_test_labels = tsv_reader.__next__()
   
        for record in tsv_reader:
            tsv_data_test.append(record)
            
        tsv_in_test.close()
        
    np_tsv_data_test = np.array(tsv_data_test)
    #prediction for testing dataset
    predicted_label_test = []
    for i in range(len(np_tsv_data_test)):
            if np_tsv_data_test[i][feature_index] == np.unique(np_tsv_data[:, feature_index])[0]:
                predicted_label_test.append(vote_group_1)
            else:
                predicted_label_test.append(vote_group_2)
                
    #metrics for testing dataset
                
    error_test = 1 - (np.sum(np_tsv_data_test[:, output_col] == predicted_label_test) / np_tsv_data_test.shape[0])
        
    #outputting
    with open(outfile_train, 'w') as f1:
        for line in predicted_label_train:
            f1.write(str(line))
            f1.write('\n')     
        f1.close()
        
    with open(outfile_test, 'w') as f2:
        for line in predicted_label_test:
            f2.write(str(line))
            f2.write('\n')   
        f2.close()
        
    with open(metrics, 'w') as f3:
            f3.write("error(train): ") 
            f3.write(str(error_train))         
            f3.write('\n')
            f3.write("error(test): ") 
            f3.write(str(error_test))
            f3.close()
        

        


