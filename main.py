from ActiveLearning import MultiViewActiveLearning
from Tools import *
from Classifer import BR_RandomForest,BR_LogisticRegression
from Evaluation import MultiLabelEvaluator
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="handle view index")
parser.add_argument('save_file',type=str)
parser.add_argument('numbers', type=int, nargs='+', help="input some integer")

args = parser.parse_args()
save_file = args.save_file
numbers = args.numbers
print(save_file,numbers,flush=True)

df_data=[]
mv_X,Y=load_data('dataset/PPTPP')
random_state_list=[42,12,1,2,3,231,34,12,412,3,521]

def signal_test(test_i):
    K_fold_dataset=train_test_split_kfold(mv_X,Y,5,random_state=random_state_list[test_i])
    k=1
    for multi_view_dataset in K_fold_dataset:
        print(f"{'*'*20}test{test_i} {k}fold{'*'*20}",flush=True)
        mval=None
        if len(numbers)==1:
            if numbers[0]==114514:
                mval=MultiViewActiveLearning(multi_view_dataset,query_statery='all-view',batch_size=30,query_ratio=0.6,view_i_list=numbers)
            else:
                mval=MultiViewActiveLearning(multi_view_dataset,query_statery='random',batch_size=30,query_ratio=0.6,view_i_list=numbers)
        else:
            mval=MultiViewActiveLearning(multi_view_dataset,query_statery='some-view',batch_size=30,query_ratio=0.6,view_i_list=numbers)
        df_data.append(mval.get_result_record())
        k+=1

for i in range(len(random_state_list)):
    signal_test(i)

df_columns=['Queries_Percent', 'AAP', 'ABP', 'ACP', 'AIP', 'AVP', 'CPP', 'PBP', 'QSP']
df_data=np.array(df_data).sum(axis=0)/(len(random_state_list)*5)
df=pd.DataFrame(df_data,columns=df_columns)
df['Queries_Percent'] = df['Queries_Percent'].apply(lambda x: f'{x * 100:.2f}%')
df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: f'{x:.4f}')
df.to_csv(f"result/csv_proba/{save_file}.csv")
