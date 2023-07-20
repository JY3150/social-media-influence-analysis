from TS.TimeSeriesBuilderBase import TimeSeriesBuilderBase
from User.UserType import UserType

from typing import List, Dict, Any, Union, Sequence
from datetime import datetime
import numpy as np


class FractionTimeSeriesConverter(TimeSeriesBuilderBase):
    """
    Fractionate the time series by Content Type.
    """
    ts_builder: TimeSeriesBuilderBase

    agg_demand: Dict[Any, List[float]]
    agg_supply: Dict[Any, List[float]]

    def __init__(self, ts_builder: TimeSeriesBuilderBase):
        self.ds = ts_builder.ds
        self.space = ts_builder.space
        self.ts_builder = ts_builder

        self.time_stamps = ts_builder.get_time_stamps()

        self.agg_demand = {}
        self.agg_supply = {}

    def create_time_series(self, user_type_or_id: Union[UserType, int],
                           content_repr: Any, mapping: str) -> List[float]:
        if mapping not in ["demand_in_community", "demand_out_community",
                           "supply"]:
            raise KeyError("Invalid Mapping Type.")
        # TODO
        print("Haven't Developed Yet")

    def create_all_type_time_series(self, user_type_or_id: Union[UserType, int],
                                    mapping: str) -> Sequence:
        raise ValueError("Invalid for Converter")

    def create_agg_time_series(self, content_repr: Any, mapping: str) -> Sequence:
        # check
        if mapping not in ["demand_in_community", "demand_out_community",
                           "supply"]:
            raise KeyError(f"Invalid Mapping Type {mapping}.")

        # 1. Check if the value is already computed
        if mapping == "demand_in_community" and content_repr in self.agg_demand.keys():
            return self.agg_demand[content_repr]
        elif mapping == "supply" and content_repr in self.agg_supply.keys():
            return self.agg_supply[content_repr]

        # 2. if the value haven't been computed, start computation
        # Step 1: gather original series
        original_dict = {}
        len_series = None
        for content_type_repr in self.space.get_all_content_type_repr():
            # extract two series
            series = None
            if mapping == "demand_in_community":
                series = self.ts_builder.create_time_series(UserType.CONSUMER, content_type_repr, mapping)
            elif mapping == "supply":
                series = self.ts_builder.create_time_series(UserType.PRODUCER, content_type_repr, mapping)
            core_node_series = self.ts_builder.create_time_series(UserType.CORE_NODE, content_type_repr, mapping)
            # add them together
            original_dict[content_type_repr] = np.add(series, core_node_series)

            # record series length
            if len_series is None:
                len_series = len(series)

        # Step 2: compute the denominator
        denom = np.zeros(len_series)
        for val in original_dict.values():
            denom += val
        # avoid division by zero
        denom[denom == 0] = 1

        # Step 3: Generate the fraction and store in class attributes
        for content_type_repr in self.space.get_all_content_type_repr():
            if mapping == "demand_in_community":
                self.agg_demand[content_type_repr] = (original_dict[content_type_repr] / denom).tolist()
            elif mapping == "supply":
                self.agg_supply[content_type_repr] = (original_dict[content_type_repr] / denom).tolist()

        # Return the output
        if mapping == "demand_in_community":
            return self.agg_demand[content_repr]
        elif mapping == "supply":
            return self.agg_supply[content_repr]
