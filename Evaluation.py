from sklearn.metrics import f1_score, roc_auc_score, average_precision_score,label_ranking_loss,roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np

class MultiLabelEvaluator:
    def __init__(self, y_true, y_pred_prob):
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob
        self.y_pred = (self.y_pred_prob >= 0.5).astype(int)

    def each_label_accuracy(self):

        return [accuracy_score(self.y_true[:, i], self.y_pred[:, i]) for i in range(self.y_true.shape[1])]

    def each_label_recall(self):

        return [recall_score(self.y_true[:, i], self.y_pred[:, i]) for i in range(self.y_true.shape[1])]

    def each_label_precision(self):

        return [precision_score(self.y_true[:, i], self.y_pred[:, i]) for i in range(self.y_true.shape[1])]

    def each_label_f1(self):

        return [f1_score(self.y_true[:, i], self.y_pred[:, i]) for i in range(self.y_true.shape[1])]
    
    def each_label_auc(self):

        return [roc_auc_score(self.y_true[:, i], self.y_pred_prob[:, i]) for i in range(self.y_true.shape[1])]
    
    def f1_macro(self):

        return f1_score(self.y_true, self.y_pred, average='macro')

    def f1_micro(self, threshold=0.5):

        return f1_score(self.y_true, self.y_pred, average='micro')

    def auc_per_label(self):

        auc_list = []
        for i in range(self.y_true.shape[1]):
            if len(np.unique(self.y_true[:, i])) > 1: 
                auc = roc_auc_score(self.y_true[:, i], self.y_pred_prob[:, i])
                auc_list.append(auc)
            else:
                auc_list.append(None) 
        return auc_list

    def auc_mean(self, average='macro'):

        num_labels = self.y_true.shape[1]

        if average == 'macro':
            auc_list = []
            for label_idx in range(num_labels):
                y_true_label = self.y_true[:, label_idx]
                y_pred_label = self.y_pred_prob[:, label_idx]

                if len(np.unique(y_true_label)) < 2:
                    continue
                
                auc = roc_auc_score(y_true_label, y_pred_label)
                auc_list.append(auc)

            return np.mean(auc_list) if auc_list else None

        elif average == 'micro':
            y_true_flat = self.y_true.ravel()
            y_pred_flat = self.y_pred_prob.ravel()
            return roc_auc_score(y_true_flat, y_pred_flat)

        else:
            raise ValueError("average must be 'macro' or 'micro'")

    def hamming_loss(self):

        num_samples, num_labels = self.y_true.shape
        mismatches = np.sum(self.y_pred != self.y_true)
        hamming_loss = mismatches / (num_samples * num_labels)
        return hamming_loss

    def avg_precision_score(self):
        average_precision_per_label = []
        for i in range(self.y_true.shape[1]):
            ap = average_precision_score(self.y_true[:, i], self.y_pred_prob[:, i])
            average_precision_per_label.append(ap)
        average_precision = np.mean(average_precision_per_label)
        return average_precision
    
    def ranking_loss(self):
        return label_ranking_loss(self.y_true, self.y_pred_prob)

    def one_error(self):

        N, L = self.y_true.shape
        one_error_count = 0

        for i in range(N):
            top_label = np.argmax(self.y_pred_prob[i])  
            if self.y_true[i, top_label] == 0:
                one_error_count += 1
        one_error = one_error_count / N
        return one_error
    
    
    
    def roc_curve_values(self):

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob)
        return np.array([fpr,tpr])

    def summary(self):


        # auc_list = self.auc_per_label()
        # for i, auc in enumerate(auc_list):
        #     print(f"AUC for Label {i}: {auc:.4f}" if auc is not None else f"AUC for Label {i}: N/A",end='\t')
        # print()
        print(f"Average_Precision_Score: {self.avg_precision_score():.4f}")
        auc_macro_value=self.auc_mean(average='macro')
        print(f"AUC_Macro_Average: {auc_macro_value:.4f}" if auc_macro_value is not None else "AUC Mean: N/A")
        auc_micro_value=self.auc_mean(average='micro')
        print(f"AUC_Micro_Average: {auc_micro_value:.4f}" if auc_micro_value is not None else "AUC Mean: N/A")
        print(f"ranking_Loss: {self.ranking_loss():.4f}")
        print(f"one_Error: {self.one_error():.4f}")

                