import torch
import torch.nn as nn

torch.manual_seed(42)
torch.cuda.empty_cache()

class RoBERTa_negation(nn.Module):
    def __init__(self, device):
        super(RoBERTa_negation, self).__init__()
        # from transformers import RobertaModel, RobertaTokenizerFast
        from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-large", num_labels=6
        ).to(device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        # set the device
        self.device = device

    def forward(self, input_ids, attention_mask, label_ids):
        """
        This function performs the forward pass.
        Args:
            input_ids: input ids
            attention_mask: attention mask
        Returns:
            outputs: outputs of the model
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=label_ids
        )
        return outputs

class DeBERTa_negation(nn.Module):
    def __init__(self, device):
        super(DeBERTa_negation, self).__init__()
        # from transformers import RobertaModel, RobertaTokenizerFast
        from transformers import DebertaV2ForSequenceClassification,  DebertaV2TokenizerFast

        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large", num_labels=6, max_position_embeddings=2048
        ).to(device)
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
        # set the device
        self.device = device

    def forward(self, input_ids, attention_mask, label_ids):
        """
        This function performs the forward pass.
        Args:
            input_ids: input ids
            attention_mask: attention mask
        Returns:
            outputs: outputs of the model
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=label_ids
        )
        return outputs