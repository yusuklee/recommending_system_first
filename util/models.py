import dataclasses
import pandas as pd
from typing import Dict, List


@dataclasses.dataclass(frozen=True) # 수정안하게끔 그리고 필드만 만들면 알아서 생성자
class Dataset:
    train:pd.DataFrame
    test:pd.DataFrame
    user_love_items:Dict[int,List[int]]
    item_content:pd.DataFrame


@dataclasses.dataclass(frozen=True)
class RecommendResult:
    rating:pd.DataFrame
    user_love_items:Dict[int,list[int]]


@dataclasses.dataclass(frozen=True)
class Metrics:
    rmse:float
    recall:float
    precision:float

    def __repr__(self):
        return f'rmse: {self.rmse:.3f}, recall: {self.recall:.3f}, precision:{self.precision:.3f}'
