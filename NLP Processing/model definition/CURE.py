class CURE(nn.Module):
    def __init__(self, target_models, target_uncertainty, predictors, hidden_dim, num_labels, num_models, dropout_ratio, freeze=True):
        super(CURE, self).__init__()
        # Hyper-parameters
        self.num_labels = num_labels
        self.num_uncertainty = len(target_uncertainty)
        self.hidden_dim = hidden_dim
        self.num_models = len(target_models)
        # Model predictions:: sub-models
        self.target_models_name = target_models
        self.target_uncertainty = target_uncertainty
        self.predictors = predictors
        # Uncertainty-aware decision fusion layers
        self.linear1 = nn.Linear(112, self.hidden_dim)
        #self.linear1 = nn.Linear(self.num_models * self.num_labels + self.num_uncertainty, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.logits = nn.Linear(self.hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(dropout_ratio)
        # Optional: Freeze sub-models
        self.reset_parameters()
        if freeze:
            for model in self.predictors:
                for name, param in model.named_parameters():
                    param.requires_grad = False

    def reset_parameters(self): # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.logits.weight)

    def make_prediction(self, model_type, predictor, input_data):
        if model_type in ["bertq"]:
            input_ids, attention_mask = input_data
            logits = predictor(input_ids, attention_mask)
            #print("bertq OK")
            return torch.sigmoid(logits)
        if model_type in ["bertc"]:
            input_ids, attention_mask, symptom = input_data
            logits = predictor(input_ids, attention_mask, symptom)
            #print("bertc OK")
            return torch.sigmoid(logits)
        if model_type in ["sympc"]:
            symptom, factor = input_data
            logits = predictor(symptom, factor)
            #print("sympc OK")
            return torch.sigmoid(logits)
        else:  # "symp"
            logits = predictor(input_data)
            #print("sympp OK")
            return torch.sigmoid(logits)

    def forward(self, text_input_ids, text_attention_mask, factor_input_ids,
                factor_attention_mask, symptom, factors, symptom_uncertainty):

        '''print(f"text_input_ids: {text_input_ids.shape}")
        print(f"text_attention_mask: {text_attention_mask.shape}")
        print(f"factor_input_ids: {factor_input_ids.shape}")
        print(f"factor_attention_mask: {factor_attention_mask.shape}")
        print(f"symptom: {symptom.shape}")
        print(f"factors: {factors.shape}")
        print(f"symptom_uncertainty: {symptom_uncertainty.shape}")'''

        # Setting variables
        #swfactor = torch.cat([symptom, factors], dim=1).unsqueeze(2)
        #symptom = symptom.unsqueeze(2).squeeze(0)
        uncertainties = []
        pred_logits = []

        # Model predictions
        with torch.no_grad():
            for model_type, predictor in zip(self.target_models_name, self.predictors):
                if model_type == "bertq":
                    input_data = (text_input_ids, text_attention_mask)
                elif model_type == "bertc":
                    input_data = (factor_input_ids, factor_attention_mask, symptom.unsqueeze(2).squeeze(0))
                elif model_type == "symp":
                    input_data = symptom.unsqueeze(2).squeeze(0)
                elif model_type == "sympc":
                    input_data = (symptom, factors)  # 기존 코드랑 다름??
                else:
                    raise ValueError("Invalid Model Type")

                pred_logit = self.make_prediction(model_type, predictor, input_data)
                pred_logits.append(pred_logit)

        # Uncertainty-aware decision fusion
        # symptom_uncertainty는 외부에서 받은 불확실성
        uncertainties.append(symptom_uncertainty.unsqueeze(1))

        '''# BertMultiLabelClassificationWithSNGP에서 얻은 불확실성
        for predictor in self.predictors:
            if isinstance(predictor, BertMultiLabelClassificationWithSNGP):
                # SNGP 모델에서 추출한 불확실성 값 받기
                mean_uncertainty = predictor.uncertainty  # 모델에서 추출한 불확실성 값
                uncertainties.append(mean_uncertainty.unsqueeze(1))'''

        uncertainties = torch.cat(uncertainties, dim=1)  # [batch size, num_uncertainty]

        stacked_all = torch.cat(pred_logits, dim=1)  # [batch_size, num_labels * num_models]
        cat = torch.cat([uncertainties.squeeze(1), stacked_all], dim=1)  # [batch_size, (num_labels * num_models) + num_uncertainty]

        hidden = self.linear1(cat)
        hidden = F.elu(self.dropout(hidden))
        hidden = self.linear2(hidden)
        hidden = F.elu(self.dropout(hidden))
        logits = self.logits(hidden)

        return logits