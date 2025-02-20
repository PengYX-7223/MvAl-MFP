import numpy as np
from math import log2
from Classifer import BR_RandomForest,BR_LogisticRegression

# batch_size=10, query_ratio=0.6, labeled_init_ratio=0.2
def crossEntropy(p1,p0):
    if p1==0 or p0==0:
        return 0
    else:
        return -((p1*log2(p1)+(p0)*log2(p0)))

def some_view_query(multi_view_dataset,batch_size=10,view_i_list=None):
    assert(view_i_list!=None)
    all_view_pred=[]
    mv_X_train,mv_X_test,Y_train,Y_test=multi_view_dataset
    test_sample_n,label_n=Y_test.shape
    
    for i in view_i_list:
        classifer=BR_LogisticRegression()
        classifer.train(mv_X_train[i],Y_train)
        all_view_pred.append(classifer.test(mv_X_test[i]))

    all_view_pred=np.array(all_view_pred)
    # print(all_view_pred)
    diff_degree=[]
    for i in range(test_sample_n):
        tot_diff=0
        for j in range(label_n):
            cnt1=0
            for k in range(len(view_i_list)):
                if all_view_pred[k][i][j]==1:
                    cnt1+=1
            each_label_diff=crossEntropy(cnt1,len(view_i_list)-cnt1)
            tot_diff+=each_label_diff
        diff_degree.append(tot_diff)
    diff_degree=np.array(diff_degree)/len(view_i_list)
    indices=diff_degree.argsort()
    #print(diff_degree[indices[:batch_size]])
    return indices[:batch_size]

def multi_view_query(multi_view_dataset,batch_size=10):
    all_view_pred=[]
    mv_X_train,mv_X_test,Y_train,Y_test=multi_view_dataset
    test_sample_n,label_n=Y_test.shape
    
    view_n=len(mv_X_train)
    
    for i in range(view_n):
        classifer=BR_LogisticRegression()
        classifer.train(mv_X_train[i],Y_train)
        all_view_pred.append(classifer.test(mv_X_test[i]))

    all_view_pred=np.array(all_view_pred)
    # print(all_view_pred)
    diff_degree=[]
    for i in range(test_sample_n):
        tot_diff=0
        for j in range(label_n):
            cnt1=0
            for k in range(view_n):
                if all_view_pred[k][i][j]==1:
                    cnt1+=1
            #each_label_diff=crossEntropy(cnt1,cnt0)
            each_label_diff=crossEntropy(cnt1,view_n-cnt1)
            tot_diff+=each_label_diff
        diff_degree.append(tot_diff)
    diff_degree=np.array(diff_degree)/view_n
    indices=diff_degree.argsort()
    #print(diff_degree[indices[:batch_size]])
    return indices[:batch_size]

import random


def random_query(sample_n,batch_size=10):
    assert(sample_n>=batch_size)
    return random.sample(range(sample_n), batch_size)
 