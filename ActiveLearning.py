from Classifer import BR_RandomForest,BR_LogisticRegression
from Evaluation import MultiLabelEvaluator
from QueryStrategy import multi_view_query,random_query,some_view_query
from Tools import *
import numpy as np
import pandas as pd
    
class MultiViewActiveLearning:
    def __init__(self, multi_view_dataset, query_ratio=0.6, batch_size=100, query_statery='random',view_i_list=None):
        mv_X_train,mv_X_test,Y_train,Y_test=multi_view_dataset
        sample_n,label_n=Y_train.shape
        view_n=len(mv_X_train)
        self.result_record=[]
        if query_statery=='all-view':
            view_i_list=[_ for _ in range(view_n)]
        
        Labeled_X, UnLabeled_X, Labeled_Y, UnLabeled_Y = Labeled_UnLabeled_split(mv_X_train,Y_train)
        init_unLabeled_num=len(UnLabeled_Y)
        init_Labeled_num=len(Labeled_Y)
        cur_sample_n=Labeled_Y.shape[0]            
        print(f"Labeled_num: {cur_sample_n}")

        mv_Y_pred=[]
        # eval
        for view_index in view_i_list:
            model=BR_LogisticRegression()
            model.train(Labeled_X[view_index],Labeled_Y)
            Y_pred=model.test(mv_X_test[view_index],proba=True)
            mv_Y_pred.append(Y_pred)
        Y_pred_result=np.array(mv_Y_pred).mean(axis=0)
        
        evaluator=MultiLabelEvaluator(Y_test, Y_pred_result)

       
        self.result_record.append(
            np.array([
                (cur_sample_n-init_Labeled_num)/init_unLabeled_num,
                evaluator.auc_mean('macro'),
                evaluator.auc_mean('micro'),
                evaluator.avg_precision_score(),
                evaluator.one_error(),
                evaluator.ranking_loss()
            ])
        )
        print(evaluator.auc_mean('macro'),
                evaluator.auc_mean('micro'),
                evaluator.avg_precision_score(),
                evaluator.one_error(),
                evaluator.ranking_loss(),
                flush=True)
        
        query_num=query_ratio*sample_n       
        
        batch_index=1
        while query_num>=batch_size:
            print(f"{'#'*10} {batch_index}th turn {'#'*10}")
            if query_statery=='random':
                indices=random_query(UnLabeled_Y.shape[0],batch_size)
            elif query_statery=='all-view':
                indices=multi_view_query((Labeled_X, UnLabeled_X, Labeled_Y, UnLabeled_Y),batch_size)
            else:
                assert(view_i_list!=None and len(view_i_list)>1)
                indices=some_view_query((Labeled_X, UnLabeled_X, Labeled_Y, UnLabeled_Y),batch_size,view_i_list)
            
            #print(cat_distr(UnLabeled_Y[indices]))
            # update unlabeled to label
            for view_index in view_i_list:
                view_Labeled = Labeled_X[view_index]
                view_UnLabeled = UnLabeled_X[view_index]

                Labeled_X[view_index] = np.vstack([view_Labeled, view_UnLabeled[indices]])
                UnLabeled_X[view_index] = np.delete(view_UnLabeled, indices, axis=0)
            
            Labeled_Y = np.vstack([Labeled_Y,UnLabeled_Y[indices]])
            UnLabeled_Y = np.delete(UnLabeled_Y,indices,axis=0)
            
            cur_sample_n=Labeled_Y.shape[0]            
            print(f"Labeled_num: {cur_sample_n}",flush=True)
            
            mv_Y_pred=[]
            # eval
            for view_index in view_i_list:
                model=BR_LogisticRegression()
                model.train(Labeled_X[view_index],Labeled_Y)
                Y_pred=model.test(mv_X_test[view_index],proba=True)
                mv_Y_pred.append(Y_pred)
            Y_pred_result=np.array(mv_Y_pred).mean(axis=0)
            
            evaluator=MultiLabelEvaluator(Y_test, Y_pred_result)
            
            self.result_record.append(
                np.array([
                    (cur_sample_n-init_Labeled_num)/init_unLabeled_num,
                    evaluator.auc_mean('macro'),
                    evaluator.auc_mean('micro'),
                    evaluator.avg_precision_score(),
                    evaluator.one_error(),
                    evaluator.ranking_loss()
                ])
            )
            print(evaluator.auc_mean('macro'),
                evaluator.auc_mean('micro'),
                evaluator.avg_precision_score(),
                evaluator.one_error(),
                evaluator.ranking_loss(),
                flush=True)
            # self.result_record.append(np.array(
            #     [(cur_sample_n-init_Labeled_num)/init_unLabeled_num]+evaluator.each_label_auc()
            # ))
            #print(self.result_record)
            # evaluator.summary()
            query_num-=batch_size
            batch_index+=1

    def get_result_record(self):
        return np.array(self.result_record)
