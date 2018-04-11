from sklearn import datasets
import numpy as np
import xlwt

class IrisAnalyzer(object):
    def __init__(self):
        # iris data
        self.iris = datasets.load_iris()
        self.data = self.iris.data
        self.target = self.iris.target

        # save results as excel tables
        self.wbk = xlwt.Workbook()
        self.char_sheet = self.wbk.add_sheet('各特征分析')
        self.coff_sheet = self.wbk.add_sheet('相关分析')
        self.metrics = {'max': np.max, 'min': np.min, 'avg': np.mean,
                   'median': np.median, 'ptp': np.ptp,
                   'std': np.std, 'var': np.var}

        # initialize the excel tables
        for i, name in enumerate(self.iris.feature_names):
            self.char_sheet.write(0, i + 1, name)
        for i, m in enumerate(self.metrics.keys()):
            self.char_sheet.write(i + 1, 0, m)

        print('----------analyzer started------------')

    def char_analysis(self):
        for i in range(self.data.shape[1]):
            char_data = self.data[:,i]
            for j, m in enumerate(self.metrics.keys()):
                self.char_sheet.write(j + 1, i + 1, self.metrics[m](char_data))

    def coff_analysis(self):
        pass

    def debug(self):
        print(self.iris.DESCR)

    def __del__(self):
        self.wbk.save('results.xls')
        print('----------results saved------------')

def debug():
    a = np.array([[1,2,3],[4,5,6]])
    print(a[:,1])

def main():
    # debug()
    ia = IrisAnalyzer()
    ia.char_analysis()
    ia.debug()

if __name__ == '__main__':
    main()