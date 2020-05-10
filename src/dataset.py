import config
import torch

class BertDataset():
    '''This is the BERT dataloader class'''

    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        #now encode the review text
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True)
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        input_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'input_type_ids': torch.tensor(input_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float)        
        }




