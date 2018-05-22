import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def apriori_gro_kline_item():
    x_sup = np.linspace(0.01, 0.1, 10)
    y_item = [332, 122, 63, 41, 31, 21, 19, 13, 10, 8]

    # start to plot
    plt.bar(x_sup, y_item, width=0.008, facecolor='#9999ff', edgecolor='white')
    # plt.bar(x + width, mean_trains, width=width,
    #         facecolor='#ffa07a', edgecolor='white', label=u'训练集')
    for x, y1 in zip(x_sup, y_item):
        plt.text(x - 0, y1 + 0, '%.2f' % y1, ha='center', va='bottom')

    plt.xlabel(u'支持度')
    plt.ylabel(u'频繁项集数')
    plt.savefig('report/img/gro_item_bar')
    plt.show()

def apriori_gro_kline_rule():
    x_con = np.linspace(0.01, 0.1, 10)
    y_rule_con_2 = [234, 73, 25, 15, 6, 2, 2, 0, 0, 0]
    y_rule_con_3 = [125, 37, 14, 7, 3, 1, 1, 0, 0, 0]
    y_rule_con_5 = [15, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x_con, y_rule_con_2, color='#90EE90', linewidth=1.7, label=u'置信度 0.2')
    ax.plot(x_con, y_rule_con_3, color='#ffa07a', linewidth=1.7, label=u'置信度 0.3')
    ax.plot(x_con, y_rule_con_5, color='#9999ff', linewidth=1.7, label=u'置信度 0.5')
    ax.scatter(x_con, y_rule_con_2, s=13, c='#90EE90')
    ax.scatter(x_con, y_rule_con_3, s=13, c='#ffa07a')
    ax.scatter(x_con, y_rule_con_5, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xticks(x_con)
    plt.xlabel(u'支持度')
    plt.ylabel(u'关联规则数')
    plt.legend()
    plt.savefig('report/img/gro_rule_kline')
    plt.show()

if __name__ == '__main__':
    apriori_gro_kline_item()