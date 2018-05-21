import csv
import random

class simData(object):

    GOODS = [
             ['bread', 'milk', 'apple', 'orange', 'beer'],          # food
             ['TV', 'PC', 'phone', 'fridge', 'ele_oven'],           # ele
             ['scissors', 'stapler', 'plate', 'knife', 'glue']      # tools
            ]

    def __init__(self, n, seed=None):
        if seed is not None:
            random.seed(seed)

        self.data = self._random_data(n)

    def _random_data(self, n):
        data = [[0 for col in range(5)] for row in range(n)]

        # 2 goods
        n_2 = n // 4
        for i in range(n_2):
            r = random.sample(range(0, 3), 1)
            random_goods = random.sample(self.GOODS[r[0]], 2)

            data[i] = random_goods

        # 3 goods
        n_3 = n_2 + n // 4
        for i in range(n_2, n_3):
            if random.random() < 0.2:
                r = random.sample(range(0, 3), 2)
                random_goods = random.sample(self.GOODS[r[0]], 2) + \
                               random.sample(self.GOODS[r[1]], 1)
            else:
                r = random.sample(range(0, 3), 1)
                random_goods = random.sample(self.GOODS[r[0]], 3)

            data[i] = random_goods

        # 4 goods
        n_4 = n_3 + n // 4
        for i in range(n_3, n_4):
            if random.random() < 0.2:
                r = random.sample(range(0, 3), 2)
                random_goods = random.sample(self.GOODS[r[0]], 3) + \
                               random.sample(self.GOODS[r[1]], 1)
            else:
                r = random.sample(range(0, 3), 1)
                random_goods = random.sample(self.GOODS[r[0]], 4)

            data[i] = random_goods

        # 5 goods
        for i in range(n_4, n):
            if i % 3 == 0:
                r = random.sample(range(0, 3), 2)
                random_goods = random.sample(self.GOODS[r[0]], 3) + \
                               random.sample(self.GOODS[r[1]], 2)
            elif i % 3 == 1:
                r = random.sample(range(0, 3), 3)
                random_goods = random.sample(self.GOODS[r[0]], 3) + \
                               random.sample(self.GOODS[r[1]], 1) + \
                               random.sample(self.GOODS[r[2]], 1)
            else:
                r = random.sample(range(0, 3), 2)
                random_goods = random.sample(self.GOODS[r[0]], 4) + \
                               random.sample(self.GOODS[r[1]], 1)

            data[i] = random_goods

        return data

    def writeCSV(self):
        csv_w = csv.writer(open('data.csv', 'w', newline=''))
        for i in range(len(self.data)):
            csv_w.writerow(self.data[i])

if __name__ == '__main__':
    sim = simData(100, 1)
    sim.writeCSV()