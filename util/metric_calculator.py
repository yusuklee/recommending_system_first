import numpy as np
from sklearn.metrics import mean_squared_error
from util.models import Metrics


class MetricCalculator:
    def calc(self, true_rating, pred_rating, user_love_items, pred_love_items,k):
        rmse = np.sqrt(mean_squared_error(true_rating, pred_rating))
        precision = self.cal_precision(user_love_items,pred_love_items,k)
        recall = self.cal_recall(user_love_items, pred_love_items,k)

        return Metrics(rmse=rmse,recall=recall,precision=precision)


    def cal_recall(self, user_love_items, pred_love_items,k):
        temp = []
        for user_id in user_love_items.keys():
            true_list = user_love_items[user_id]
            pred_list = pred_love_items[user_id][:k]
            rate = len(set(true_list)&set(pred_list))/len(true_list)
            temp.append(rate)
        return np.mean(temp)

    def cal_precision(self, user_love_items, pred_love_items, k):
        temp = []
        for user_id in user_love_items.keys():
            true_list = user_love_items[user_id]
            pred_list = pred_love_items[user_id][:k]
            rate = len(set(true_list) & set(pred_list)) / k
            temp.append(rate)
        return np.mean(temp)
