from transformers import RobertaModel, BertModel
import torch
import torch.nn as nn

class TextAndContextEncoder(torch.nn.Module):
    def __init__(self,model='bert-base-uncased',num_labels = 3,single_modalitity=False, device="cpu"):
        super().__init__()

        self.bert_model = BertModel.from_pretrained(model).to(device)
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        self.dropout = nn.Dropout(p=0.4)

        # If True a classifier is added at the end
        self.single_modalitity = single_modalitity
        if self.single_modalitity:
            size_context_emb = 50
            size_utterence_emb = 100
            self.fusion = [nn.Linear(768, size_context_emb).to(device), nn.Linear(768, size_context_emb).to(device),
                           nn.Linear(768, size_utterence_emb).to(device)]
            self.linear = nn.Linear(size_utterence_emb + 2 * size_context_emb, size_utterence_emb).to(device)
            self.classifier = nn.Linear(size_utterence_emb, num_labels).to(device)

    def forward(self, text,attention_mask):

        for i in range(3):
            o = self.bert_model(text[:, i], attention_mask=attention_mask[:, i]).last_hidden_state

            if i==0:
                x = o.unsqueeze(1)
            else:
                x = torch.cat([x,o.unsqueeze(1)],dim=1)

        # x is of shape (batch_size, 3, sequence_length, bert_hidden_size)

        if self.single_modalitity:
            for i in range(3):
                x[:, i,attention_mask[:, i] == 0] = 0
                x[:, i] = x[:, i].sum(dim=1)
                x[:, i] = x[:, i] / (attention_mask[:, i]).sum(dim=1).unsqueeze(-1).expand_as(x[:, i])
                x[:, i] = self.fusion31(x[:, i])
            x = torch.cat((x[:, 0], x[:, 1], x[:, 2]), dim=1)
            x = self.linear(x)
            x = self.classifier(x)
            return x
        else:
            return x[:, 0], x[:, 1], x[:, 2]


class TextEncoder(torch.nn.Module):
    def __init__(self,model='bert-base-uncased',num_labels = 3,single_modalitity=False,device="cpu"):
        super().__init__()

        self.bert_model = BertModel.from_pretrained(model).to(device)
        embedding_size=100
        #If True a classifier is added at the end
        self.single_modalitity = single_modalitity


        self.fusion =nn.Linear(768, embedding_size).to(device)

        if self.single_modalitity:
            self.classifier = nn.Linear(embedding_size, num_labels).to(device)

        self.softmax = nn.Softmax(dim=1)
        self.device = device

        self.dropout = nn.Dropout(p=0.4)


    def forward(self, x,attention_mask):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.text_model(x[:, 2], attention_mask=attention_mask[:, 2]).last_hidden_state



        if self.single_modalitity:
            x[attention_mask[:, 2] == 0] = 0
            x = x.sum(dim=1)
            x = x / (attention_mask[:, 2]).sum(dim=1).unsqueeze(-1).expand_as(x)
            x = self.fusion(x)


            x = self.classifier(x)
        return x


