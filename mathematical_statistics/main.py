import pandas as pd
import statsmodels.api as sm


def looper(limit,data):
    cols = ['x1','x2','x3','x4','x5','x6','x7','x8']
    for i in range(len(cols)):
        data1 = data[cols]
        x = sm.add_constant(data1)  #生成自变量
        y = data['y']   #生成因变量
        model = sm.OLS(y,x)     #生成模型
        result = model.fit()    #模型拟合

        pvalues = result.pvalues    #得到结果中所有P值
        pvalues.drop('const',inplace=True)  #把const去掉

        pmax = max(pvalues) #选出最大的P值

        if pmax > limit:
            index = pvalues.idxmax()
            cols.remove(index)
        else:
            return result


if __name__ == '__main__':
    file_path = r'my_data.csv'
    data = pd.read_csv(file_path)
    data.columns = ['y','x1','x2','x3','x4','x5','x6','x7','x8']
    # print(data)
    # x = sm.add_constant(data.iloc[:,1:])
    # y = data['y']
    # model = sm.OLS(y,x)
    # result = model.fit()
    result = looper(0.05,data)  #最优回归方程分析
    print(result.summary())





