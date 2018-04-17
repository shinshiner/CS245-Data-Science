from sklearn import datasets
import stats
import numpy as np
import xlwt

class IrisAnalyzer(object):
    def __init__(self):
        # iris data
        self.iris = datasets.load_iris()
        self.data = self.iris.data
        self.target = self.iris.target
        print(self.iris.DESCR)

        # save results as excel tables
        self.wbk = xlwt.Workbook()
        self.char_sheet = self.wbk.add_sheet('各特征分析')
        self.corr_sheet = self.wbk.add_sheet('相关分析')
        self.metrics = {'max': np.max, 'min': np.min, 'avg': np.mean,
                    'median': np.median, 'ptp': np.ptp,
                    'std': np.std, 'var': np.var, 'q1': stats.quantile,
                    'q3': stats.quantile}

        # initialize the excel tables
        for i, name in enumerate(self.iris.feature_names):
            self.char_sheet.write(0, i + 1, name)
        for i, m in enumerate(self.metrics.keys()):
            self.char_sheet.write(i + 1, 0, m)
        for i, name in enumerate(self.iris.feature_names):
            self.corr_sheet.write(0, i + 1, name)
        for i, m in enumerate(self.iris.feature_names):
            self.corr_sheet.write(i + 1, 0, m)
        self.corr_sheet.write(i + 2, 0, 'target')

        print('----------analyzer started------------')

    def run(self):
        self.basic_att()
        self.char_analysis()
        self.corr_analysis()
        
        self.wbk.save('results.xls')
        print('----------results saved------------')

    def basic_att(self):
        with open('basic_att.txt', 'w') as f:
            f.write(self.iris.DESCR + '\n\n')

            f.write('Type of data: ' + str(type(self.data)) + '\n\n')
            f.write('Shape of data: ' + str(self.data.shape) + '\n\n')
            f.write('Feature names: ' + str(self.iris.feature_names) + '\n\n')

            f.write('Target: ' + str(self.target) + '\n\n')
            f.write('Type of target: ' + str(type(self.target)) + '\n\n')
            f.write('Shape of target: ' + str(self.target.shape) + '\n\n')
            f.write('Target names: ' + str(self.iris.target_names) + '\n\n')

    def char_analysis(self):
        for i in range(self.data.shape[1]):
            char_data = self.data[:,i]
            for j, m in enumerate(self.metrics.keys()):
                if not m in ['q1', 'q3']:
                    self.char_sheet.write(j + 1, i + 1, self.metrics[m](char_data))
                elif m == 'q1':
                    self.char_sheet.write(j + 1, i + 1, self.metrics[m](char_data, p=0.25))
                elif m == 'q3':
                    self.char_sheet.write(j + 1, i + 1, self.metrics[m](char_data, p=0.75))

    def corr_analysis(self):
        for i in range(self.data.shape[1]):
            char_data = self.data[:, i]

            # corr among characters
            for j in range(self.data.shape[1]):
                self.corr_sheet.write(j + 1, i + 1, np.corrcoef(char_data, self.data[:, j])[0][1])

            # corr between characters and target
            self.corr_sheet.write(j + 2, i + 1, np.corrcoef(char_data, self.target)[0][1])

def debug():
    pass

if __name__ == '__main__':
    ia = IrisAnalyzer()
    ia.run()