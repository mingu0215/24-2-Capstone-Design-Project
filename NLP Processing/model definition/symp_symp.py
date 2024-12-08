# 모델 정의
class Symp(nn.Module):
    def __init__(self, input_dim, filter_num=128, filter_sizes=[1], dropout=0.2, num_labels=16, k_max_pooling=1):
        super(Symp, self).__init__()
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, filter_num, size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_labels)
        self.k_max_pooling = k_max_pooling

    def forward(self, input_seqs):
        # Conv1d 적용 및 활성화 함수
        x = [F.relu(conv(input_seqs)) for conv in self.convs]

        # k-max pooling 적용
        x = [self.kmax_pooling(item, self.k_max_pooling) for item in x]

        # 평균값 계산 및 결합
        x = [item.mean(dim=2) for item in x]  # [batch_size, filter_num]
        x = torch.cat(x, 1)  # [batch_size, len(filter_sizes) * filter_num]

        # 드롭아웃 및 최종 출력
        x = self.dropout(x)
        logits = self.fc(x)  # [batch_size, num_labels]
        return torch.sigmoid(logits)  # Sigmoid 활성화 함수 적용


    def kmax_pooling(self, x, k):
        return x.sort(dim=2, descending=True)[0][:, :, :k]