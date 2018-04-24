import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']    # display Chinese
plt.rcParams['axes.unicode_minus'] = False      # display minus sign
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

import os

class bostonAnalyzer(object):
    def __init__(self):
        self.dataset = datasets.load_boston()
        self.data = self.dataset.data
        self.target = self.dataset.target
        self.var = []

        if not os.path.exists('report/img'):
            os.makedirs('report/img')

    def run(self, vis=False):
        self.var.clear()
        with open('report/result.txt', 'w') as f:
            for i in range(13):
                pca_op = PCA(n_components=i)
                pca_res = pca_op.fit_transform(self.data)
                self.var.append(pca_op.explained_variance_ratio_.sum())

                f.write('###### Dimension %d ######\n' % i)
                f.write(str(pca_res.shape) + '\n')
                f.write(str(pca_op.explained_variance_ratio_) + '\n')
                f.write(str(pca_op.explained_variance_ratio_.sum()) + '\n\n')

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
        # print(self.data.shape)
        # print(self.target.shape)
        self._k_line()

    def _k_line(self):
        x = list(range(len(self.var)))
        plt.scatter(x, self.var, s=14, c='r')
        plt.plot(x, self.var)
        plt.xlabel('主成分个数')
        plt.ylabel('降维后各特征方差比例之和')
        plt.savefig('report/img/kline')
        plt.show()

if __name__ == '__main__':
    bA = bostonAnalyzer()
    bA.run(True)
    # bA.show()