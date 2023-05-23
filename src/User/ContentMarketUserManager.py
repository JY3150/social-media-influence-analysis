from User.ContentMarketUser import ContentMarketUser
from User.ContentMarketConsumer import ContentMarketConsumer
from User.ContentMarketProducer import ContentMarketProducer
from User.ContentMarketCoreNode import ContentMarketCoreNode
from UserPartitioning.UsersStrategy import UsersStrategy
from DAO.ContentMarketDAO import ContentMarketDAO
from User.UserType import UserType
from Tweet.TweetType import TweetType
from Tweet.ContentMarketTweet import ContentMarketTweet
from Clustering.ContentMarketClustering import ContentMarketClustering
from ContentSpace.ContentSpace import ContentSpace

from typing import Set, Dict, Any, List
from datetime import datetime


class ContentMarketUserManager:
    # Attributes
    consumers: Set[ContentMarketConsumer]
    producers: Set[ContentMarketProducer]
    core_nodes: Set[ContentMarketCoreNode]

    def __init__(self, dao: ContentMarketDAO, partition: UsersStrategy):
        print("=================Build Users=================")
        # initialize variables
        self.consumers = set()
        self.producers = set()
        self.core_nodes = set()

        # build users
        users = self._build_users(dao)
        print("=================Partition Users=================")
        self._partition_users(users, partition)

        print("=========Successfully Build UserManager=========")

    def _build_users(self, dao: ContentMarketDAO) -> Set[ContentMarketUser]:
        users = set()
        for user in dao.load_community_users():
            user_dict = {
                "user_id": user["userid"],
                "rank": user["rank"],
                "username": user["username"],
                "influence_one": user["influence one"],
                "influence_two": user["influence two"],
                "production_utility": user["production utility"],
                "consumption_utility": user["consumption utility"],
                "local_follower_count": user["local follower"],
                "local_following_count": user["local following"],
                "local_followers": user["local follower list"],
                "local_following": user["local following list"],
                "global_follower_count": user["global follower"],
                "global_following_count": user["global following"],
                "is_new_user": user["is new user"]
            }
            new_user = ContentMarketUser(**user_dict)
            users.add(new_user)
        return users

    def _partition_users(self, users: Set[ContentMarketUser],
                         partition: UsersStrategy) -> None:
        """Split <users> into consumers, producers, and core nodes
        by <partition>, and store them into class variables.
        """
        for user in users:
            if user.rank < 10:  # 10 top users are core nodes
                core_node = ContentMarketCoreNode(**vars(user))
                self.core_nodes.add(core_node)
            else:
                # the rest ordinary user are consumer/producer
                if partition.is_producer(user):
                    new_prod = ContentMarketProducer(**vars(user))
                    self.producers.add(new_prod)
                if partition.is_consumer(user):
                    new_consumer = ContentMarketConsumer(**vars(user))
                    self.consumers.add(new_consumer)

    def get_user(self, userid: int) -> ContentMarketUser:
        """Return a User with <userid>.
        """
        for user_group in [self.consumers, self.producers, self.core_nodes]:
            for user in user_group:
                if user.user_id == userid:
                    return user

        # if this is not for any user, raise Exception
        raise Exception(f"`{userid}` is not in the list")

    def get_type_users(self, user_type: UserType) -> Set[ContentMarketUser]:
        """Return a set of all users of Type <user_type>.
        """
        if user_type == UserType.CONSUMER:
            return self.consumers
        elif user_type == UserType.PRODUCER:
            return self.producers
        elif user_type == UserType.CORE_NODE:
            return self.core_nodes
        else:
            raise Exception(f"Invalid User Type `{user_type}`")

    def add_tweet(self, tweet: ContentMarketTweet,
                  tweet_type: TweetType) -> None:
        """Add <tweet> with <tweet_type> to the user by <tweet.get_userid()>.
        """
        if tweet_type == TweetType.ORIGINAL_TWEET:
            user = self.get_user(tweet.user_id)
            user.original_tweets.add(tweet)
        elif tweet_type == TweetType.QUOTE_OF_IN_COMM:
            user = self.get_user(tweet.user_id)
            user.quotes_of_in_community.add(tweet)
        elif tweet_type == TweetType.QUOTE_OF_OUT_COMM:
            user = self.get_user(int(tweet.quote_user_id))
            user.quotes_of_out_community.add(tweet)
        elif tweet_type == TweetType.RETWEET_OF_IN_COMM:
            user = self.get_user(tweet.user_id)
            user.retweets_of_in_community.add(tweet)
        elif tweet_type == TweetType.RETWEET_OF_OUT_COMM:
            user = self.get_user(int(tweet.retweet_user_id))
            user.retweets_of_out_community.add(tweet)
        else:
            raise Exception(f"Invalid Tweet Type `{tweet_type}` when adding")

    def get_user_tweets(self, userid, tweet_type: TweetType):
        """Return a list of Tweet for user with <userid> of
        type <tweet_type>.
        """
        user = self.get_user(userid)
        if tweet_type == TweetType.ORIGINAL_TWEET:
            return user.original_tweets
        elif tweet_type == TweetType.QUOTE_OF_IN_COMM:
            return user.quotes_of_in_community
        elif tweet_type == TweetType.QUOTE_OF_OUT_COMM:
            return user.quotes_of_out_community
        elif tweet_type == TweetType.RETWEET_OF_IN_COMM:
            return user.retweets_of_in_community
        elif tweet_type == TweetType.RETWEET_OF_OUT_COMM:
            return user.retweets_of_out_community
        else:
            raise Exception(f"Invalid Tweet Type `{tweet_type}` when getting")

    def calculate_mapping(self, clustering: ContentMarketClustering) -> None:
        """Summarize supply and demand information in each user (regardless of
        creation time period).
        """
        # Consumers
        print("=================Calculate Consumer Mapping=================")
        for consumer in self.consumers:
            consumer.calculate_demand(clustering)

        # Producers
        print("=================Calculate Producer Mapping=================")
        for producer in self.producers:
            producer.calculate_supply(clustering)

        # Core Nodes
        print("================Calculate Core Node Mapping================")
        for core_node in self.core_nodes:
            core_node.calculate_demand(clustering)
            core_node.calculate_supply(clustering)

    def calculate_time_mapping(self, user_type: UserType,
                               start_time: datetime, end_time:datetime,
                               clustering: ContentMarketClustering,
                               content_space: ContentSpace,
                               tweet_types: List[TweetType],
                               tweet_manager) -> Dict[Any, int]:
        # TODO: resolve this circular import issue
        # initialize dictionary storage
        freq_dict = {}
        for content_type in content_space.get_all_content_types():
            freq_dict[content_type.get_representation()] = 0

        # accumulate time series
        for user in self.get_type_users(user_type):
            for tweet_type in tweet_types:
                for tweet in self.get_user_tweets(user.user_id, tweet_type):
                    # This is bad practice: shouldn't cross boundary
                    freq_dict[clustering.get_content_type(tweet.id).get_representation()] \
                        += tweet_manager.tweet_created_between_time(tweet.id, start_time, end_time)

        # return dictionary
        return freq_dict
