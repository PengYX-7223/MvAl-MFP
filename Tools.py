import numpy as np
import os
from sklearn.model_selection import KFold

LEBALS_NAME=['AAP', 'ABP', 'ACP', 'AIP', 'AVP', 'CPP', 'PBP', 'QSP']

def load_data(root):
    files=os.listdir(root)
    files.sort()
    mv_X=[]
    print(files)
    for file in files:
        file_path=os.path.join(root,file)
        if file=='labels.txt':
            Y=np.loadtxt(file_path)
        else:
            X=np.loadtxt(file_path)
            mv_X.append(X)
    return mv_X,Y

def stratified_sample(X_train, Y_train, label_percentage=0.05, total_percentage=0.1, random_state=42):
    np.random.seed(random_state)
    num_samples = Y_train.shape[0]
    num_labels = Y_train.shape[1]
    
    selected_indices = set()

    for label in range(num_labels):
        label_indices = np.where(Y_train[:, label] == 1)[0]
        if len(label_indices) > 0:
            sample_size = max(1, int(len(label_indices) * label_percentage))
            selected_indices.update(np.random.choice(label_indices, size=sample_size, replace=False))
    
    selected_indices = list(selected_indices)
    remaining_samples = max(1, int(num_samples * total_percentage) - len(selected_indices))

    remaining_indices = np.setdiff1d(np.arange(num_samples), selected_indices)
    if len(remaining_indices) > 0:
        additional_indices = np.random.choice(remaining_indices, size=remaining_samples, replace=False)
        selected_indices.extend(additional_indices)

    return np.array(selected_indices)

def Labeled_UnLabeled_split(X_train,Y_train,label_percentage=0.025, total_percentage=0.05,random_state=42):
    n_sample=Y_train.shape[0]
    
    Labeled_indices=set(stratified_sample(X_train,Y_train,label_percentage,total_percentage))
    UnLabeled_indices=set(range(0,n_sample))-Labeled_indices

    Labeled_indices=list(Labeled_indices)
    UnLabeled_indices=list(UnLabeled_indices)

    Labeled_X = [view[Labeled_indices] for view in X_train]
    UnLabeled_X = [view[UnLabeled_indices] for view in X_train]
    
    Labeled_Y = Y_train[Labeled_indices]
    UnLabeled_Y = Y_train[UnLabeled_indices]
    
    return Labeled_X, UnLabeled_X, Labeled_Y, UnLabeled_Y

def train_test_split(X,Y,test_size=0.2,random_state=42):
    n_sample=Y.shape[0]
    indices = np.arange(n_sample)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    test_size = int(n_sample * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    X_train = [view[train_indices] for view in X]
    X_test = [view[test_indices] for view in X]
    
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    
    return X_train, X_test, Y_train, Y_test

def train_test_split_kfold(X, Y, n_splits=5,random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True,random_state=random_state)
    splits = []
    for train_indices, test_indices in kf.split(Y):
        X_train = [view[train_indices] for view in X]
        X_test = [view[test_indices] for view in X]
        
        Y_train = Y[train_indices]
        Y_test = Y[test_indices]
        
        splits.append((X_train, X_test, Y_train, Y_test))
    
    return splits
def cat_distr(labels):
    n,m=labels.shape
    tot=labels.sum()
    result={}
    for j in range(m):
        temp=0
        for i in range(n):
            if labels[i][j]==1:
                temp+=1
        result[LEBALS_NAME[j]]=round(temp/tot,4)
    return result

if __name__=='__main__':
    labels=np.array([
        [0,1,0,0, 0,1,0,0],
        [0,1,0,0, 1,0,0,0],
        [0,0,0,0, 1,0,0,0],
        [0,0,1,0, 1,0,0,0],
        [0,0,0,0, 1,0,0,0],
        [1,0,0,0, 0,0,0,0],

    ])
    print(cat_distr(labels))
    