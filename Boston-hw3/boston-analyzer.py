import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']    # display Chinese
plt.rcParams['axes.unicode_minus'] = False      # display minus sign
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

import os
import time

class bostonAnalyzer(object):
    def __init__(self):
        # load dataset
        self.dataset = datasets.load_boston()
        self.data = self.dataset.data
        self.target = self.dataset.target

        # for plot
        self.var = []
        self.t = []

        if not os.path.exists('report/img'):
            os.makedirs('report/img')

    def run(self, vis=False):
        '''
        Run PCA for 1-13 dimensions
        :param vis: True: plot dim 1-3; False: not plot
        '''

        self.var.clear()
        self.t.clear()

        with open('report/result.txt', 'w') as f:
            for i in range(13):
                t = time.time()
                pca_op = PCA(n_components=i)
                pca_res = pca_op.fit_transform(self.data)
                t = time.time() - t

                self.var.append(pca_op.explained_variance_ratio_.sum())
                self.t.append(t)

                # write log
                f.write('###### Dimension %d ######\n' % i)
                f.write(str(pca_res.shape) + '\n')
                f.write(str(pca_op.explained_variance_ratio_) + '\n')
                f.write(str(pca_op.explained_variance_ratio_.sum()) + '\n')
                f.write(str(t) + '\n\n')

                # visualize for debug & plot
                if vis:
                    if i == 1:
                        # plot 1 dimension
                        plt.scatter(pca_res[:, 0], pca_res[:, 0], s=14, c=self.target)
                        plt.savefig('report/img/pca-%d' % i)
                        plt.show()
                    elif i == 2:
                        # plot 2 dimensions
                        plt.scatter(pca_res[:,0], pca_res[:,1], s=8, c=self.target)
                        plt.savefig('report/img/pca-%d' % i)
                        plt.show()
                    elif i == 3:
                        # plot 3 dimensions
                        ax = plt.subplot(projection='3d')
                        ax.scatter(pca_res[:, 0], pca_res[:, 1], pca_res[:, 2], s=8, c=self.target)
                        plt.savefig('report/img/pca-%d' % i)
                        plt.show()

    def show(self):
        '''
        Show the basic info of Boston dataset & plot k-lines
        '''
        print(self.data.shape)
        print(self.target.shape)
        self._k_line_radio()
        self._k_line_time()

    def _k_line_radio(self):
        '''
        Plot k-line of variant radio
        '''
        x = list(range(len(self.var)))
        plt.scatter(x, self.var, s=14, c='r')
        plt.plot(x, self.var)
        plt.xlabel('主成分个数')
        plt.ylabel('降维后各特征方差比例之和')
        plt.savefig('report/img/kline-radio')
        plt.show()

    def _k_line_time(self):
        '''
        Plot k-line of time
        '''
        x = list(range(len(self.t)))
        plt.scatter(x, self.t, s=14, c='r')
        plt.plot(x, self.t)
        plt.xlabel('主成分个数')
        plt.ylabel('PCA算法消耗时间(s)')
        plt.savefig('report/img/kline-time')
        plt.show()

if __name__ == '__main__':
    bA = bostonAnalyzer()
    bA.run(True)
    bA.show()