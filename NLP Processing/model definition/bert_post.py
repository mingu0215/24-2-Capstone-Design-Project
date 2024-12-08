# PsyEx (only risky-post stream) 모델 정의
class PsyExOnlyPostStream(nn.Module):
    def __init__(self, model_name, num_labels, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_name = model_name
        self.num_heads = num_heads
        self.num_labels = num_labels
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.num_labels)])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.num_labels)])

    def forward(self, input_ids, attention_mask):
        """
        Args:
            batch: A dictionary containing the following keys:
                - input_ids: Tensor of shape [batch_size, seq_len]
                - attention_mask: Tensor of shape [batch_size, seq_len]
                - token_type_ids: Tensor of shape [batch_size, seq_len] (optional)
        Returns:
            logits: Tensor of shape [batch_size, num_labels]
        """
        input_ids = input_ids  # [batch_size, seq_len]
        attention_mask = attention_mask  # [batch_size, seq_len]
        #token_type_ids = batch.get("token_type_ids", None)  # Optional

        # BERT Encoder
        post_outputs = self.post_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
            #token_type_ids=token_type_ids
        )

        # Pooling 처리
        if self.pool_type == "first":
            x = post_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        elif self.pool_type == 'mean':
            x = self.mean_pooling(post_outputs.last_hidden_state, attention_mask)  # [batch_size, hidden_dim]

        # Transformer Encoder
        x = x + self.pos_emb[:x.size(0), :]  # Add positional embeddings
        x = self.user_encoder(x.unsqueeze(1)).squeeze(1)  # [batch_size, hidden_dim]

        # Attention & Classification
        logits = []
        for i in range(self.num_labels):
            logits.append(self.clf[i](x))  # [batch_size, 1]
        logits = torch.cat(logits, dim=1)  # [batch_size, num_labels]

        return logits

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Mean pooling using attention mask.
        Args:
            last_hidden_state: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        Returns:
            Pooled output: [batch_size, hidden_dim]
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask