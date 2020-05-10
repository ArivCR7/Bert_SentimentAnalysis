import config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import dataset
import torch
from model import BertBaseUncased
from transformers import AdamW, get_linear_schedule_with_warmup
import engine
import pandas as pd
import numpy as np

def run():
    dfx = pd.read_csv(config.TRAINING_FILE)
    dfx.sentiment = dfx.sentiment.apply(
        lambda x: 1 if x == 'positive' else 0
    )
    df_train, df_val = train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    train_dataset = dataset.BertDataset(
        df_train.review.values,
        df_train.sentiment.values
        )
    
    val_dataset = dataset.BertDataset(
        df_val.review.values,
        df_val.sentiment.values
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers = 1
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers = 4
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = BertBaseUncased()
    model = model.to(device)

    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_params, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    )

    best_accuracy = 0
    accumulation_steps = config.ACCUMULATION 
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, scheduler, device, accumulation_steps)
        preds, actuals = engine.eval_fn(val_dataloader, model, device)
        preds = np.array(preds) >= 0.5
        accuracy = metrics.accuracy_score(actuals, preds)
        print("Accuracy Score: %0.2f"%(accuracy))

        if accuracy > best_accuracy:
            print("Best Accuracy reached, saving model...")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == '__main__':
    run()
