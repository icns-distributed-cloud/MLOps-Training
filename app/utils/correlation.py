import pandas as pd
import scipy.stats as stats

import os


if __name__ == '__main__':
    correlation_list = []
    cor01 = pd.DataFrame(pd.read_csv('data/data.csv'))
    cols = cor01.columns
    for i in range(len(cols)-2):
        corr = stats.pearsonr(cor01[cols[1]], cor01[cols[i+2]])
        correlation_list.append(corr[0])

    print(stats.pearsonr(cor01[cols[1]], cor01[cols[1]]))
    print(correlation_list)



    # cor01.columns = ['date', 'sales', 'year', 'month', 'hour', 'day', 'lowtemp', 'hightemp']

    


