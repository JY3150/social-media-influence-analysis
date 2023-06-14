from User.UserManager import UserManager
from User.UserType import UserType
from Tweet.TweetType import TweetType
from Aggregation.AggregationBase import AggregationBase
from Mapping.ContentType import ContentType
from Tweet.MinimalTweet import MinimalTweet

from typing import Dict, List, Any, Set, DefaultDict
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict


def _merge_dicts(dict1: Dict[Any, Set[MinimalTweet]], dict2: Dict[Any,
                 Set[MinimalTweet]]) -> None:
    """Update dict1.
    """
    for key, value in dict2.items():
        dict1[key].update(value)


class ContentDemandSupply(AggregationBase):
    """Aggregate Supply and Demand Information for time series processing.
    """
    # Attributes
    name: str
    content_space: Set[ContentType]
    user_manager: UserManager

    demand: Dict[UserType, DefaultDict[Any, Set[MinimalTweet]]]
    supply: Dict[UserType, DefaultDict[Any, Set[MinimalTweet]]]

    def __init__(self, *args):
        # create()
        # param: str, Set[ContentType], UserManager, TweetManager
        if len(args) == 5:
            super().__init__(args[0], args[2], args[3])
            # load from arguments
            self.content_space = args[1]
            self.user_manager = args[2]

            # initialize demand and supply
            self.demand = {UserType.CONSUMER: defaultdict(set),
                           UserType.CORE_NODE: defaultdict(set)}
            self.supply = {UserType.CORE_NODE: defaultdict(set),
                           UserType.PRODUCER: defaultdict(set)}
        # load()
        # param: str, Set[ContentType],
        #        Dict[UserType, Dict[Any, Set[MinimalTweet]]],
        #        Dict[UserType, Dict[Any, Set[MinimalTweet]]]
        elif len(args) == 4:
            self.name = args[0]
            self.content_space = args[1]
            self.demand = args[2]
            self.supply = args[3]

    def _create_time_stamps(self, start: datetime, end: datetime,
                            period: timedelta) -> None:
        """Create a list of time stamps for partitioning the Tweet, and
        store in self.time_stamps.
        """
        # TODO: move to new module
        curr_time = start
        while curr_time <= end:
            self.time_stamps.append(curr_time)
            curr_time += period

    def clear_trailing_zero(self) -> None:
        # TODO: move to new class
        pass

    # # Below are methods for extraction from outer space
    # def get_type_demand_series(self, user_type: UserType) -> Dict[Any, np.array]:
    #     """Return the demand time series for <user_type>.
    #     """
    #     # get users
    #     users = self.user_manager.get_type_users(user_type)
    #
    #     # get data
    #     new_dict = defaultdict(lambda: np.array([0 for _ in
    #                                              range(len(self.time_stamps))]))
    #     for user in users:
    #         curve = self.demand[user.user_id]
    #         for key, value in curve.items():
    #             new_dict[key] += np.array(value)
    #     return new_dict
    #
    # def get_type_supply_series(self, user_type: UserType) -> Dict[Any, np.array]:
    #     """Return the supply time series for <user_type>.
    #     """
    #     # get users
    #     users = self.user_manager.get_type_users(user_type)
    #
    #     # get data
    #     new_dict = defaultdict(lambda: np.array([0 for _ in
    #                                              range(len(self.time_stamps))]))
    #     for user in users:
    #         curve = self.supply[user.user_id]
    #         for key, value in curve.items():
    #             new_dict[key] += np.array(value)
    #     return new_dict
    #
    # def get_agg_demand(self, user_type: UserType) -> Dict[Any, int]:
    #     """Return the aggregate demand dictionary for <user_type>.
    #     """
    #     # TODO
    #     pass
    #
    # def get_agg_supply(self, user_type: UserType) -> Dict[Any, int]:
    #     """Return the aggregate supply dictionary for <user_type>.
    #     """
    #     # TODO
    #     pass
    #
    # def get_agg_type_demand_series(self, user_type: UserType) -> np.array:
    #     demand_series = list(self.get_type_demand_series(user_type).values())
    #     return np.array(demand_series).sum(axis=0)
    #
    # def get_agg_type_supply_series(self, user_type: UserType) -> np.array:
    #     supply_series = list(self.get_type_supply_series(user_type).values())
    #     return np.array(supply_series).sum(axis=0)
    #
    # def get_content_type_repr(self) -> List:
    #     return [content_type.get_representation() for content_type
    #             in self.content_space]

    ##########################################################
    # Test Version
    ##########################################################
    def _calculate_user_type_mapping(self, user_type: UserType,
                                     storage: Dict[UserType,
                                     Dict[Any, Set[MinimalTweet]]],
                                     tweet_types: List[TweetType]) -> None:
        for user in tqdm(self.user_manager.get_type_users(user_type)):
            # ignore this type warning
            freq_dict = self.user_manager.calculate_user_time_mapping(
                user, tweet_types)
            _merge_dicts(storage[user_type], freq_dict)

    def calculate_demand(self):
        print("Start User Demand")
        demand_spec = [TweetType.RETWEET_OF_IN_COMM,
                       TweetType.RETWEET_OF_OUT_COMM]
        self._calculate_user_type_mapping(UserType.CONSUMER, self.demand,
                                          demand_spec)
        self._calculate_user_type_mapping(UserType.CORE_NODE, self.demand,
                                          demand_spec)

    def calculate_supply(self):
        print("Start User Supply")
        supply_spec = [TweetType.ORIGINAL_TWEET]
        self._calculate_user_type_mapping(UserType.PRODUCER, self.supply,
                                          supply_spec)
        self._calculate_user_type_mapping(UserType.CORE_NODE, self.supply,
                                          supply_spec)
