import numpy as np

from Aggregation.ContentSpace import ContentSpace
from TS.TimeSeriesBuilder import TimeSeriesBuilder
from User.UserType import UserType
from Causality.CausalityAnalysisTool import *

from typing import List, Dict


class AggregateCausalityAnalysis:
    # Attributes
    space: ContentSpace
    ts_builder: TimeSeriesBuilder
    lags: List[int]

    supply: List[int]
    demand: List[int]

    def __init__(self, space: ContentSpace, ts_builder: TimeSeriesBuilder,
                 lags: List[int]):
        self.space = space
        self.ts_builder = ts_builder
        self.lags = lags

        # Calculate Supply and Demand
        demand = self.ts_builder.create_all_type_time_series(UserType.CONSUMER,
                                                             "demand_in_community")
        self.demand = np.add(demand, self.ts_builder.create_all_type_time_series(UserType.CORE_NODE,
                                                                                 "demand_in_community"))

        supply = self.ts_builder.create_all_type_time_series(UserType.PRODUCER,
                                                             "supply")
        self.supply = np.add(supply, self.ts_builder.create_all_type_time_series(
            UserType.CORE_NODE, "supply"))

    def do_granger_cause(self) -> Dict[int, float]:
        result = {}
        for lag in self.lags:
            if lag < 0:
                result[lag] = gc_score_for_lag(self.supply, self.demand, -lag)
            elif lag > 0:
                result[lag] = gc_score_for_lag(self.demand, self.supply, lag)
            else:
                result[lag] = 1
        return result

    def do_granger_cause_agg_to_user(self, user_id: int) -> Dict[int, float]:
        # Calculate Individual Supply
        supply = self.ts_builder.create_all_type_time_series(user_id, "supply")

        # Do Causality
        result = {}
        for lag in self.lags:
            if lag < 0:
                result[lag] = gc_score_for_lag(supply, self.demand, -lag)
            elif lag > 0:
                result[lag] = gc_score_for_lag(self.demand, supply, lag)
            else:
                result[lag] = 1
        return result

    def do_cos_similarity(self) -> Dict[int, float]:
        output = cs_for_lags(self.demand, self.supply, self.lags)
        return dict(zip(self.lags, output))

    def do_cos_similarity_agg_to_user(self, user_id: int) -> Dict[int, float]:
        # Calculate Individual Supply
        supply = self.ts_builder.create_all_type_time_series(user_id, "supply")

        # Calculate similarity
        output = cs_for_lags(self.demand, supply, self.lags)
        return dict(zip(self.lags, output))
