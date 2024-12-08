class IntegratedModelWithBERT(nn.Module):
    def __init__(self, bert_model_name, embedding_dim, symptom_dim, num_labels, filter_num=64, filter_sizes=[1, 3, 5], max_pooling_k=1, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.symptom_dim = symptom_dim
        self.num_labels = num_labels

        # BERT 모델 초기화
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # BERT의 출력 크기

        # CNN Layers
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.convs = nn.ModuleList([nn.Conv1d(2, filter_num, size) for size in filter_sizes])  # input_channels=2

        self.max_pooling_k = max_pooling_k
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size + self.bert_hidden_size, num_labels)  # BERT 출력과 CNN 출력 결합

    def forward(self, factor_input_ids, factor_attention_mask, symptoms):
        # BERT 출력 계산
        bert_outputs = self.bert(input_ids=factor_input_ids, attention_mask=factor_attention_mask)
        bert_pooled_output = bert_outputs.pooler_output  # (batch_size, bert_hidden_size)

        # 차원 확장: factor_input_ids에 마지막 차원 추가
        factor_input_ids = factor_input_ids.unsqueeze(-1).float()  # (batch_size, seq_len, 1)

        # Pooling으로 차원 축소: seq_len → symptom_dim
        factors_reduced = F.adaptive_avg_pool1d(factor_input_ids.transpose(1, 2), output_size=self.symptom_dim).transpose(1, 2)  # (batch_size, symptom_dim, 1)=(64,48,1)

        # 증상 데이터 차원 맞춤
        input_seqs = torch.cat([symptoms, factors_reduced], dim=2)  # (batch_size, symptom_dim, 2)

        # Conv1d 입력 차원 변환
        input_seqs = input_seqs.transpose(1, 2)  # (batch_size, 2, symptom_dim)

        # CNN 처리
        x = [F.relu(conv(input_seqs)) for conv in self.convs]
        x = [self.kmax_pooling(item, self.max_pooling_k).mean(2) for item in x]
        x = torch.cat(x, 1)  # (batch_size, hidden_size)
        x = self.dropout(x)

        # BERT 출력과 CNN 출력을 결합
        combined = torch.cat([bert_pooled_output, x], dim=1)  # (batch_size, hidden_size + bert_hidden_size)

        # Fully Connected Layer for prediction
        logits = self.fc(combined)  # (batch_size, num_labels)
        return logits

    def kmax_pooling(self, x, k):
        if x.dim() < 3:
            raise ValueError(f"Input to kmax_pooling must have at least 3 dimensions, but got {x.shape}")
        return x.sort(dim=2, descending=True)[0][:, :, :k]