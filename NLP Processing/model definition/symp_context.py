# 모델 정의
class SympWithFactor(nn.Module):
    def __init__(self, input_dim, num_factors, filter_num=64, filter_sizes=[1], dropout=0.2, num_labels=16, max_pooling_k=1):
        super(SympWithFactor, self).__init__()
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.max_pooling_k = max_pooling_k
        self.convs = nn.ModuleList([nn.Conv1d(16, filter_num, size) for size in filter_sizes]) #num_factors * 2->16
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_labels)

    def forward(self, symptoms, factors):
        # symptoms를 축소하여 factors와 동일한 크기로 조정
        symptoms_reduced = self.kmax_pooling(symptoms.unsqueeze(1), k=factors.size(1)).squeeze(1)  # [batch_size, num_factors]

        # 축소된 symptoms와 factors를 결합
        input_seqs = torch.cat([symptoms_reduced, factors], dim=1).unsqueeze(1)  # [batch_size, 1, num_factors * 2]

        # Conv1d에 맞게 차원 전환
        input_seqs = input_seqs.transpose(1, 2)  # [batch_size, num_factors * 2, 1]
        x = [F.relu(conv(input_seqs)) for conv in self.convs]
        x = [self.kmax_pooling(item, self.max_pooling_k).mean(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def kmax_pooling(self, x, k):
        return x.sort(dim=2, descending=True)[0][:, :, :k]