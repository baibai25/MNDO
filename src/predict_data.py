import re
import pandas as pd
from imblearn import metrics

# calc evaluate metrics
def calc_metrics(y_test, pred, auc, i):
        sen_macro = metrics.sensitivity_score(y_test, pred, pos_label=1, average='macro')   
        sen_micro = metrics.sensitivity_score(y_test, pred, pos_label=1, average='micro')    
        spe_macro = metrics.specificity_score(y_test, pred, pos_label=1, average='macro')   
        spe_micro = metrics.specificity_score(y_test, pred, pos_label=1, average='micro')
        geo_macro = metrics.geometric_mean_score(y_test, pred, pos_label=1, average='macro')   
        geo_mico = metrics.geometric_mean_score(y_test, pred, pos_label=1, average='micro')
        index = ['sm', 'b1', 'b2', 'enn', 'tom', 'ada', 'mnd'] 
        metrics_list = [index[i], sen_macro, sen_micro, spe_macro, spe_micro, geo_macro, geo_mico, auc]
        return metrics_list

# convert classification report to dataframe
def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)
