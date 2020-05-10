import transformers
import torch.nn as nn
import config

class BertBaseUncased(nn.Module):
    '''This is the bert model class'''
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids, 
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        bo = self.bert_dropout(o2)
        output = self.out(bo)
        return output

        