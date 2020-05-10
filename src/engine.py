from tqdm import tqdm
import torch.nn as nn
import torch

def loss_fn(preds, actuals):
    return nn.BCEWithLogitsLoss()(preds, actuals.view(-1, 1))

def train_fn(data_loader, model, optimizer, scheduler, device, accumulation_steps):
    
    model.train()
    i = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        
        torch.cuda.empty_cache() 
        ids = d['ids']
        mask = d['mask']
        input_type_ids = d['input_type_ids']     
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        input_type_ids = input_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(
            ids,
            mask,
            input_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        del ids, mask, input_type_ids, targets
        torch.cuda.empty_cache() 
        if i % accumulation_steps == 0:
            print("Training loss after iter: %d is %0.2f"%(i, loss.item()))

        i+=1 

def eval_fn(data_loader, model, device):
    
    model.eval()
    fin_outputs = []
    fin_targets = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids']
            mask = d['mask']
            input_type_ids = d['input_type_ids']     
            targets = d['targets']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            input_type_ids = input_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)


            outputs = model(
                ids,
                mask,
                input_type_ids
            )
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets  
