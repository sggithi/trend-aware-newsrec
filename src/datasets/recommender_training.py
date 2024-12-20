import random
from collections import defaultdict
from typing import Union

import pandas as pd
from torch.utils.data import Dataset

from src.utils.data import load_behaviors, load_news
from src.utils.encode import CategoricalEncoder
from src.utils.tokenize import NltkTokenizer, PLMTokenizer


def filter_positive_samples(impressions: list[str]) -> list[str]:
    return [sample[:-2] for sample in impressions if sample.endswith("-1")]


def filter_negative_samples(impressions: list[str]) -> list[str]:
    return [sample[:-2] for sample in impressions if sample.endswith("-0")]


Tokenizer = Union[NltkTokenizer, PLMTokenizer]
TokenizerOutput = Union[list[int], dict[str, list[int]]]
NewsItem = dict[str, Union[TokenizerOutput, int]]


class RecommenderTrainingDataset(Dataset):
    """
    Dataset for training recommenders with negative sampling.
    Returns logs of history and sampled impressions.
    """

    def __init__(
        self,
        mind_variant: str,
        tokenizer: Tokenizer,
        negative_sampling_ratio: int = 4,
        num_words_title: int = 20,
        num_words_abstract: int = 50,
        history_length: int = 50,
        news_features: list[str] = ["title"],
        categorical_encoders: dict[str, CategoricalEncoder] = defaultdict(
            CategoricalEncoder
        ),
    ):
        self.mind_variant = mind_variant
        self.split = "train"
        self.tokenizer = tokenizer
        self.negative_sampling_ratio = negative_sampling_ratio
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.history_length = history_length
        self.news_features = news_features
        self.categorical_encoders = categorical_encoders

        print("Loading news...")
        self.news = self.prepare_news()

        print("Loading logs...")
        self.logs = self.prepare_logs()

    @property
    def num_words(self) -> int:
        return self.tokenizer.vocab_size + 1

    @property
    def num_categories(self) -> int:
        return self.categorical_encoders["category"].n_categories + 1

    @property
    def num_subcategories(self) -> int:
        return self.categorical_encoders["subcategory"].n_categories + 1

    def prepare_logs(self) -> pd.DataFrame:
        behaviors = load_behaviors(self.mind_variant, splits=[self.split])
        
        # Split impressions into positive and negative samples
        behaviors["positive_samples"] = behaviors.impressions.apply(
            filter_positive_samples
        )
        behaviors["negative_samples"] = behaviors.impressions.apply(
            filter_negative_samples
        )

        # Filter out entries with too few negative samples
        behaviors = behaviors[
            behaviors.negative_samples.map(len) >= self.negative_sampling_ratio
        ]

        # Create one datapoint for every positive sample
        behaviors = behaviors.explode("positive_samples").rename(
            columns={"positive_samples": "positive_sample"}
        )
        behaviors.negative_samples = behaviors.negative_samples.apply(
            lambda x: random.sample(x, k=self.negative_sampling_ratio)
        )
        behaviors["candidate_news"] = (
            behaviors.positive_sample.apply(lambda x: [x]) + behaviors.negative_samples
        )
        behaviors["time"] = pd.to_datetime(behaviors["time"])
        behaviors = behaviors.sort_values("time")
        max_impressions = max(behaviors.impressions.map(len))
    
        behaviors.impressions = behaviors.impressions.apply(
        lambda x: x + ["N0-0"] * (max_impressions - len(x))
    )
        behaviors = behaviors.reset_index(drop=True).drop(
            columns=[
                #"time",
                "log_id",
                #"impressions",
                #"positive_sample",
                #"negative_samples",
            ]
        )

        return behaviors

    def prepare_news(self) -> dict[str, NewsItem]:
        textual_features = ["title", "abstract"]

        news = load_news(
            self.mind_variant, splits=[self.split], columns=self.news_features
        )
        parsed_news: dict[str, NewsItem] = {}
        for index, row in news.iterrows():
            article: NewsItem = {}
            for feature in self.news_features:
                if feature in textual_features:
                    article[feature] = self.tokenizer(
                        row[feature].lower(),
                        getattr(self, f"num_words_{feature}"),
                    )
                if feature == "category":
                    article[feature] = self.categorical_encoders[feature].encode(
                        row["category"]
                    )
                if feature == "subcategory":
                    article[feature] = self.categorical_encoders[feature].encode(
                        (row["category"], row["subcategory"])
                    )
            parsed_news[str(index)] = article

        return parsed_news

    def pad_history(self, history: list[NewsItem]) -> tuple[list[NewsItem], list[int]]:
        padding_all = {
            "title": self.tokenizer("", self.num_words_title),
            "abstract": self.tokenizer("", self.num_words_abstract),
            "category": 0,
            "subcategory": 0,
        }
        padding = {feature: padding_all[feature] for feature in self.news_features}
        padding_length = self.history_length - len(history)
        padded_history = [padding] * padding_length + history
        mask = [0] * padding_length + [1] * len(history)
        return padded_history, mask

    def __len__(self) -> int:
        return len(self.logs)

    def __getitem__(self, idx: int) -> tuple[list[NewsItem], list[int], list[NewsItem]]:
        row = self.logs.iloc[idx]
        ##############################
        answer = 0
        candidates = [0] * 300
        negative_samples = [0] * 4
        if idx != 0:
            behaviors_row = self.logs.iloc[idx - 1]
            answer = int(behaviors_row["positive_sample"][1:])
            candidates = [int(item[1:-2]) for item in behaviors_row["impressions"]]
            negative_samples = [int(item[1:]) for item in behaviors_row["negative_samples"]]
        time = row["time"]
        ###############################
        history, mask = self.pad_history(
            [self.news[id] for id in row.history[-self.history_length :]]
        )
        candidate_news = [self.news[id] for id in row.candidate_news]
        time = time.timestamp()
        return history, mask, candidate_news, answer, candidates, negative_samples, candidates, time
