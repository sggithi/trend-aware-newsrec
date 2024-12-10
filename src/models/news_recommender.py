import torch
import torch.nn as nn

from src.utils.utils import object_to_device
from collections import Counter
import pandas as pd
from collections import defaultdict


class NewsRecommender(nn.Module):
    def __init__(self, news_encoder, user_encoder, click_predictor, loss_modules, ctr_predictor):
        super(NewsRecommender, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.click_predictor = click_predictor
        self.ctr_predictor = ctr_predictor  # CTR 예측 모델 추가

        if isinstance(loss_modules, list):
            self.loss_modules = nn.ModuleList(loss_modules)
        else:
            self.loss_modules = nn.ModuleList([loss_modules])

        self.pass_features = (
            hasattr(self.user_encoder, "requires_features")
            and self.user_encoder.requires_features
        )
        self.news_clicks = Counter()  # 클릭된 횟수 저장
        self.news_impressions = Counter()  
        self.beta = nn.Parameter(torch.tensor(0.2))  # learnable parameter
        self.popularity_predictor = nn.Linear(1, 1)
        self.news_ctr_history =  defaultdict(list)
        self.max_history_len = 50

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_news(self, news):
        news = object_to_device(news, self.device)
        news_vectors = self.news_encoder(news)
        if not self.pass_features:
            return news_vectors
        news["vectors"] = news_vectors
        return news

    def encode_user(self, clicked_news, mask=None):
        clicked_news = object_to_device(clicked_news, self.device)
        if mask is not None:
            return self.user_encoder(clicked_news, mask.to(self.device))
        return self.user_encoder(clicked_news)

    def rank(self, candidate_news, user_vector):
        return self.click_predictor(candidate_news, user_vector)

    ##################################################################
    def update_ctr(self, answer, candidates):
        if type(answer) == int:
            self.news_clicks[answer] += 1
        else:
            for answers in answer:
                if answers == 0:
                    continue
                self.news_clicks[answers] += 1
                
        
        if isinstance(candidates[0], list):  # Check if the first element is a list
            candidates = [item for sublist in candidates for item in sublist]
        for impression in candidates:
            if impression == 0:
                continue
            self.news_impressions[impression] += 1
            
    def get_ctr(self, news_id):
        """
        Compute the CTR for a given news item.
        """
        if self.news_ctr_history[news_id]:
            return self.news_ctr_history[news_id][-1]
        elif self.news_impressions[news_id] > 0:
            return self.news_clicks[news_id] / self.news_impressions[news_id]
        else:
            return 0  # If no impressions, return 0 as CTR
    
    def train_ctr_lstm(self, news_id):
        # LSTM을 학습시키기 위한 준비
        ctr_sequence = self.news_ctr_history[news_id][:-1] # exclude last item (answer) 
        if len(ctr_sequence) < 2:  # 최소 두 개의 데이터 포인트가 있어야 학습 가능
            return None, None

        # CTR 시퀀스를 tensor로 변환
        ctr_tensor = torch.tensor(ctr_sequence, dtype=torch.float32).unsqueeze(-1)  # [seq_len, 1]

        # LSTM에 입력할 수 있도록 배치 차원을 추가
        ctr_tensor = ctr_tensor.unsqueeze(0)  # [1, seq_len, 1]

        # LSTM을 통해 예측된 CTR 값 얻기
        predicted_ctr = self.ctr_predictor_lstm(ctr_tensor)
        return predicted_ctr
    ##################################################################

    def forward(self, candidate_news, clicked_news, labels, answer, candidates, negative_samples, history_ids, time, mask=None):
        #candidate_news_repr = self.encode_news(candidate_news)
        #clicked_news_repr = self.encode_news(clicked_news)

        ##########################
        lstm_loss_fn = nn.MSELoss()
        answer_ctr_list = []
        lstm_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        lstm_loss_count = 0
        
        for answer_id in answer:
            if isinstance(answer_id, torch.Tensor):
                answer_id = answer_id.item()
            answer_ctr = self.news_ctr_history[answer_id]
           
            if len(answer_ctr) > 1:  # 최소 2개 이상 있어야 학습 가능
                answer_ctr_tensor = torch.tensor(answer_ctr[:-1]).unsqueeze(0).unsqueeze(-1).to(self.device)  # [seq_len, 1]
                next_ctr = torch.tensor(answer_ctr[-1]).unsqueeze(-1).to(self.device)  # Ground truth next CTR
                #print("answer_ctr", answer_ctr_tensor.tolist(), answer_ctr_tensor.shape)
                predicted_next_ctr = self.ctr_predictor(answer_ctr_tensor)  # LSTM 예측
                #if answer_id ==9935:
                #    print("predicted_next_ctr", predicted_next_ctr, "next_ctr", next_ctr)
                # 손실 계산
                lstm_loss += lstm_loss_fn(predicted_next_ctr.float(), next_ctr.float())
                lstm_loss_count += 1
                lstm_loss= lstm_loss.to(torch.float32)

                answer_ctr_list.append(predicted_next_ctr)
            else:
                if len(answer_ctr) > 0:
                    answer_ctr_tensor = torch.tensor(answer_ctr).unsqueeze(0).unsqueeze(-1).to(self.device)  # [seq_len, 1]
                    predicted_next_ctr = self.ctr_predictor(answer_ctr_tensor)
                    answer_ctr_list.append(predicted_next_ctr)
                # 히스토리가 없으면 기본값 추가
                else:
                    answer_ctr_list.append(torch.zeros(1, 1).to(self.device))

        #ctr_answer_predicted = torch.tensor(answer_ctr_list).unsqueeze(-1).to(self.device) 

        ctr_negative_samples_predictions = [] 
        for candidate_list in negative_samples:
            ctr_negative_pred_list = []
            for nws_id in candidate_list:
                ctr_negative_sample_sequence = torch.tensor(self.news_ctr_history[nws_id]).to(self.device)
                
                if len(ctr_negative_sample_sequence) > 1:
                    seq_tensor = torch.tensor(ctr_negative_sample_sequence[:-1]).unsqueeze(-1).to(self.device)  # [seq_len, 1]
                    next_ctr = torch.tensor(ctr_negative_sample_sequence[-1]).unsqueeze(-1).to(self.device)  # Ground truth next CTR
                    predicted_next_ctr = self.ctr_predictor(seq_tensor)

                    lstm_loss += lstm_loss_fn(predicted_next_ctr, next_ctr)
                    lstm_loss_count += 1

                    ctr_negative_pred_list.append(predicted_next_ctr)  # LSTM 예측된 CTR
                else:
                    ctr_negative_pred = torch.zeros(1, 1).to(self.device)  # Default to 0 if no history
                    ctr_negative_pred_list.append(ctr_negative_pred)
    
            ctr_negative_samples_predictions.append(ctr_negative_pred_list)
            
        ctr_negative_samples_predictions = torch.tensor(ctr_negative_samples_predictions).to(self.device) # [batch_size]
        if lstm_loss_count > 0:
            lstm_loss /= lstm_loss_count
        ##############################
        total_loss = lstm_loss
        return total_loss
