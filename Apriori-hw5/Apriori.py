from itertools import chain, combinations
from collections import defaultdict
import numpy as np
import time
from simData import *

class Apriori(object):
    def __init__(self, f_name, sup=0.1, con=0.1):
        self.data = self._read_csv(f_name)
        self.sup = sup
        self.con = con
        self.items = []
        self.rules = []

    def _read_csv(self, f_name):
        with open(f_name, 'r') as f:
            for line in f:
                line = line.strip().rstrip(',')
                item = frozenset(line.split(','))
                yield item

    def run(self):
        # 小工具函数
        def _get_support(item):
            return float(freq_set[item]) / len(deals)
        def _get_subsets(item):
            return chain(*[combinations(item, i + 1) for i, a in enumerate(item)])
        def _join_set(item_set, length):
            return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])

        # 初始化 L1 项集和交易数据
        item_set, deals = self._init_data()

        freq_set = defaultdict(int)
        large_set = dict()

        init_set = self._remove_item_set(item_set, deals, freq_set)
        l_set = init_set

        k = 2
        while (l_set != set([])):
            large_set[k - 1] = l_set
            l_set = _join_set(l_set, k)
            c_set = self._remove_item_set(l_set, deals, freq_set)
            l_set = c_set
            k += 1

        # 组合结果
        for key, value in large_set.items():
            self.items.extend([(tuple(item), _get_support(item)) for item in value])

        for key, value in list(large_set.items())[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in _get_subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        con = _get_support(item) / _get_support(element)
                        if con >= self.con:
                            self.rules.append(((tuple(element), tuple(remain)), con))

    def _init_data(self):
        deal_list = list()
        item_set = set()

        for d in self.data:
            deal = frozenset(d)
            deal_list.append(deal)

            # 产生 L1 项集
            for item in deal:
                item_set.add(frozenset([item]))

        return item_set, deal_list

    # 根据支持度阈值删除项集
    def _remove_item_set(self, item_set, deals, freq_set):
        res = set()
        local_set = defaultdict(int)

        # 统计
        for item in item_set:
            for d in deals:
                if item.issubset(d):
                    freq_set[item] += 1
                    local_set[item] += 1

        # 删除
        for item, count in local_set.items():
            support = float(count) / len(deals)
            if support >= self.sup:
                res.add(item)

        return res

    # 输出结果
    def show(self):
        print(u'------频繁项集------')
        for item, sup in sorted(self.items, key=lambda items: items[1]):
            # print ('%s , %.2f' % (str(item), sup))
            try:
                print('%s, %s & %.2f \\\\' % (str(item[0]), str(item[1]), sup))
            except:
                print('%s & %.2f \\\\' % (str(item[0]), sup))
        print (u'\n------关联规则------')
        for rule, con in sorted(self.rules, key=lambda rules: rules[1]):
            pre, post = rule
            print ('%s --> %s , %.2f' % (str(pre), str(post), con))

        return len(self.items), len(self.rules)

def force_items():
    t = time.time()
    def power_sets_binary(items):
        N = len(items)
        for i in range(2**N):
            combo = []
            for j in range(N):
                if (i >> j) % 2 == 1:
                    combo.append(items[j])
            yield combo

    all_items = ['bread', 'milk', 'apple', 'orange', 'beer',
                 'TV', 'PC', 'phone', 'fridge', 'ele_oven',
                 'scissors', 'stapler', 'plate', 'knife', 'glue']
    subsets = list(power_sets_binary(all_items))
    counts = [0] * len(subsets)
    cnt = 0

    with open('data.csv', 'r') as f:
        for line in f:
            item = line.strip().rstrip(',').split(',')
            for i, s in enumerate(subsets):
                if set(s).issubset(set(item)) or set(s) == set(item):
                    counts[i] += 1
                    cnt += 1

    return time.time() - t

if __name__ == "__main__":
    # x = np.linspace(0.01, 0.1, 10)
    # it = []
    # ru = []
    # for i in range(x.shape[0]):
    #     print(i, '!!!!!!!!!!!!!!!!!!!')
    #     a = Apriori('groceries.csv', x[i], 0.5)
    #     a.run()
    #     itt, ruu = a.show()
    #     it.append(itt)
    #     ru.append(ruu)
    # print(it)
    # print(ru)

    a = Apriori('groceries.csv', 0.05, 0.2)
    a.run()
    i, r = a.show()
    print(i, r)

    # t_f = []
    # t_a = []
    # for i in range(100, 501, 50):
    #     sim = simData(i, 1)
    #     sim.writeCSV()
    #     tmp = force_items()
    #     t_f.append(tmp)
    #
    #     t = time.time()
    #     a = Apriori('data.csv', 0.15, 0.6)
    #     a.run()
    #     t_a.append(time.time() - t)
    #
    # print(t_f)
    # print(t_a)