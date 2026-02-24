from abc import ABC,abstractmethod

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset, **kwargs):
        pass

    def eval(self):
        dataset = DataLoader(user_size=1000, test_size=5, data_path="ml-10M100K").main()
        #데이터 셋에는 train,test, user2items, movies가 존재하고
        pred = self.recommend(dataset)

        metrics = MetricCalculator().calc(
            true_rating=dataset.test.rating.tolist(),
            pred_rating=pred.rating.tolist(),
            user_love_items=dataset.user_love_items,
            pred_love_items=pred.user_love_items,
            k=10,
        )
        print(metrics)
