from multiprocessing import Pool
from typing import Union

import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.behaviors import BehaviorsDataset, behaviors_collate_fn
from src.datasets.news import NewsDataset
from src.evaluation.metrics import mrr_score, ndcg_score
from src.models.news_recommender import NewsRecommender
from src.utils.collate import collate_fn
from src.utils.encode import CategoricalEncoder
from src.utils.tokenize import NltkTokenizer, PLMTokenizer
from src.utils.utils import add_first_dim, get_user_repr_from_index, object_to_device

TokenizerOutput = Union[list[int], dict[str, list[int]]]


scoring_functions = {
    "AUC": roc_auc_score,
    "MRR": mrr_score,
    "NDCG@5": lambda y_true, y_score: ndcg_score(y_true, y_score, 5),
    "NDCG@10": lambda y_true, y_score: ndcg_score(y_true, y_score, 10),
}


def calculate_metrics(result):
    for metric, scoring_fn in scoring_functions.items():
        result[metric] = scoring_fn(result["clicked"], result["probs"])
    return result


def evaluate(
    model: NewsRecommender,
    split: str,
    tokenizer: Union[NltkTokenizer, PLMTokenizer],
    categorical_encoders: dict[str, CategoricalEncoder],
    cfg: DictConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    tokenizer.eval()
    for encoder in categorical_encoders.values():
        encoder.eval()

    news_dataset = NewsDataset(
        cfg.data.mind_variant,
        split,
        tokenizer,
        cfg.num_words_title,
        cfg.num_words_abstract,
        categorical_encoders,
        cfg.features,
    )
    news_dataloader = DataLoader(
        news_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )
    news_vectors = {}

    # Precompute news vectors
    with torch.no_grad():
        for news_ids, batched_news_features in tqdm(
            news_dataloader,
            desc="Encoding news for evaluation",
        ):
            output = model.encode_news(batched_news_features)
            output = object_to_device(output, torch.device("cpu"))
            for i, id in enumerate(news_ids):
                if isinstance(output, dict):
                    news_vectors[id] = {key: value[i] for key, value in output.items()}
                else:
                    news_vectors[id] = output[i]

    behaviors_dataset = BehaviorsDataset(cfg.data.mind_variant, split, news_vectors)
    behaviors_dataloader = DataLoader(
        behaviors_dataset,
        batch_size=cfg.batch_size,
        collate_fn=behaviors_collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # Make predictions
    results = []
    time_interval  = 2 * 60 * 60 
    last_reset_time = None
    with torch.no_grad():
        for log_ids, clicked_news_vectors, mask, impression_ids, clicked, answer, history_ids, time in tqdm(
            behaviors_dataloader, desc="Evaluating logs"
        ):
            if cfg.use_history_mask:
                user_vectors = model.encode_user(clicked_news_vectors, mask)
            else:
                user_vectors = model.encode_user(clicked_news_vectors)
            
            current_time = time[0]  # time은 Tensor일 수 있음, .item()으로 값 추출

            for i in range(len(log_ids)):
                impressions = object_to_device(
                    add_first_dim(
                        collate_fn([news_vectors[id] for id in impression_ids[i]])
                    ),
                    device,
                )
                user_repr = get_user_repr_from_index(user_vectors, i)

                probs = model.rank(impressions, user_repr).squeeze(0)
                impressions = [int(impression_id[1:]) for impression_id in impression_ids[i]]
                model.update_ctr(answer[i], impressions)
              
                # LSTM을 통해 answer의 CTR 예측
                ctr_predictions = []
                for impression_id in impression_ids[i]:
                    ctr_seq = model.news_ctr_history[int(impression_id[1:])]

                    ctr_seq = torch.tensor(ctr_seq).unsqueeze(-1).to(model.device)
                    if len(ctr_seq) > 0:  # Ensure there's a sequence to process
                        ctr_sequence = ctr_seq.unsqueeze(0)  # Add batch dimension [1, seq_len, 1]
                        ctr_predicted = model.ctr_predictor(ctr_sequence)  # LSTM CTR
                    else:
                        ctr_predicted = torch.zeros(1, 1).to(model.device)  # Default to 0 if no history
         
                    ctr_predictions.append(ctr_predicted)
                         
    
            if last_reset_time is None:
                last_reset_time = current_time 
            else:
                time_diff = current_time - last_reset_time 
            
                if time_diff >= time_interval: 
                    combined_keys = set(model.news_impressions.keys()).union(model.news_ctr_history.keys())
                    for impression in combined_keys: 
                        if isinstance(impression, torch.Tensor):
                            impression = impression.item()
                        if model.news_impressions[impression] == 0:
                            ctr = 0
                        else:
                            ctr = model.news_clicks[impression] / model.news_impressions[impression]
               
                        if len(model.news_ctr_history[impression]) >= model.max_history_len:
                            model.news_ctr_history[impression].pop(0) 
                        model.news_ctr_history[impression].append(ctr) 
                  
                    model.news_clicks.clear()
                    model.news_impressions.clear()

                    last_reset_time = current_time

                ctr_predictions_tensor = torch.tensor(ctr_predictions).to(model.device)
                
                adjusted_probs = probs * (1 - model.beta) + model.beta * ctr_predictions_tensor
       
                probs_list = adjusted_probs.tolist()
                results.append(
                    {"log_id": log_ids[i], "clicked": clicked[i], "probs": probs_list}
                )

    # Calculate metrics
    with Pool(processes=cfg.num_workers) as pool:
        scores = pool.map(calculate_metrics, results)

    eval_data = pd.DataFrame(scores)
    eval_data = eval_data.set_index("log_id")

    metrics = {metric: eval_data[metric].mean() for metric in scoring_functions}
    probs = eval_data["probs"].reset_index()

    return metrics, probs
