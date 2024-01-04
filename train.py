
import torch
import torch.nn as nn
import torch.optim as optim
from malefic  import  MALEFIC
from accelerate import Accelerator
from tqdm import tqdm
from AudioEncoder import AudioEncoder
from AUEncoder import AUEncoder
from TextEncoder import TextEncoder
from torchmetrics.classification import MulticlassF1Score
from utils import label_to_target
from data import create_dataset

def train(name,malefic=True,model_drop_audio=False,model_drop_context=False,model_drop_face = False,model_drop_text=False,embracement_size=300):

    train_dataloader, validation_dataloader = create_dataset(dataset_path="data")

    epochs = 50
    lr = 5e-6
    best_vallos = 100000
    bestF1=0
    accelerator = Accelerator()
    device = accelerator.device
    torch.manual_seed(2302)
    model_single_modality = (model_drop_audio + model_drop_face + model_drop_text + model_drop_context )== 3
    if model_single_modality:
        if not model_drop_text:
            model = TextEncoder('bert-base-uncased',single_modalitity=True)
        if not model_drop_face:
            model  = AudioEncoder(model_single_modality=True)
        if not model_drop_audio:
            model = AUEncoder(model_single_modality=True)
    else:
        model = MALEFIC(device,True,malefic=malefic,model_drop_audio=model_drop_audio,model_drop_face=model_drop_face,model_drop_text=model_drop_text,embracement_size=embracement_size)
    classification_loss = nn.CrossEntropyLoss()
    F1_metric = MulticlassF1Score(num_classes=label_to_target['change']+1,average='macro')

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs, 1)

    model, optimizer, train_dataloader,validation_dataloader,F1_metric,scheduler = accelerator.prepare(model, optimizer, train_dataloader,validation_dataloader,F1_metric,scheduler)

    for epoch in range(epochs):

        loss = 0
        valloss = 0
        classloss = 0
        total_F1=0
        valF1=0
        valclassloss = 0
        model.train()
        trues = None
        for batch_features in tqdm(train_dataloader):

            target = batch_features[2].to(device)
            prosody = batch_features[0].to(device)
            pad_prosody = batch_features[1].to(device)
            text = batch_features[3].to(device)
            attention = batch_features[4].to(device)
            action_units = batch_features[5].to(device).type(torch.float)
            pad_au = batch_features[6].to(device)
            present_face = batch_features[7].to(device)

            optimizer.zero_grad()

            # compute reconstructions
            if model_single_modality:
                if not model_drop_text:
                    classe = model(text,attention)
                if not model_drop_face:
                    classe = model(action_units,action_units)
                if not model_drop_audio:
                    classe = model(prosody,pad_prosody)
            else:
                classe,weight = model(text,attention,prosody,pad_prosody,action_units,pad_au,present_face)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            train_loss = classification_loss(classe, target.to(int))
            F1_score = F1_metric(classe, target.to(int))

            # compute accumulated gradients
            accelerator.backward(train_loss)
            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            total_F1 += F1_score.item()
            classloss += train_loss.item()
            scheduler.step()
        # compute the epoch training loss
        loss = loss / len(train_dataloader)
        total_F1 = total_F1/len(train_dataloader)
        #Evaluation loop
        model.eval()
        for batch_features in tqdm(validation_dataloader):

            target = batch_features[2]
            prosody = batch_features[0]
            pad_prosody = batch_features[1]
            text = batch_features[3]
            attention = batch_features[4]
            action_units = batch_features[5].type(torch.float)
            pad_au = batch_features[6]
            present_face = batch_features[7]

            if model_single_modality:
                if not model_drop_text:
                    classe = model(text, attention)
                if not model_drop_face:
                    classe = model(action_units,pad_au)
                if not model_drop_audio:
                    classe = model(prosody,pad_prosody)
            else:
                classe,weight = model( text, attention,prosody, pad_prosody, action_units, pad_au, present_face)

            val_loss = classification_loss(classe, target.to(int))
            if trues is None:
                preds = classe.detach()
                trues  =target
            else:
                preds = torch.cat((preds, classe.detach()), dim=0)
                trues = torch.cat((trues, target), dim=0)

            # add the mini-batch training loss to epoch loss
            valloss += val_loss.item()

        valloss = valloss / len(validation_dataloader)
        valF1 = F1_metric(torch.tensor(preds).to(device), torch.tensor(trues).to(device).to(int)).item()
        if valloss < best_vallos:
            try:
                torch.save(model, name)
            except:
                pass
            print("model saved with best val loss")
            best_vallos = valloss
        if valF1 > bestF1:
            try:
                torch.save(model, name+"_F1")
            except:
                pass
            print("model saved with best F1")
            bestF1 = valF1
        # scheduler.step(valloss)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f},F1= {:.6f},val loss=  {:.6f}".format(epoch + 1, epochs, loss,valF1,valloss))

