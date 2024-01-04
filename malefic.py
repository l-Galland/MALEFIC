

import torch
import torch.nn as nn

from embracenet.embracenet import EmbraceNet
from AudioEncoder import AudioEncoder
from TextEncoder import TextAndContextEncoder
from AUEncoder import  AUEncoder

class MALEFIC(nn.Module):
    def __init__(self, device, is_training, model_drop_audio=False, model_drop_text=False, model_dropout=True,
                 model_drop_face=False, embracement_size=300,all_att=False,malefic=True):
        super(MALEFIC, self).__init__()

        self.device = device
        self.n_modalities = 5
        self.is_training = is_training
        self.model_drop_audio = model_drop_audio
        self.model_drop_text = model_drop_text
        self.model_drop_face = model_drop_face
        self.au_embedding_size = 256
        self.audio_embedding_size = 768
        self.text_embedding_size= 768
        self.text_context_embedding_size = 768
        self.embracement_size =embracement_size
        self.model_dropout = model_dropout

        # check if only one modality is not dropped
        self.single_modalitie = (model_drop_audio+model_drop_text+model_drop_face)==2
        self.text_encoder = TextAndContextEncoder()
        self.audio_encoder = AudioEncoder()
        self.au_encoder = AUEncoder()
        self.dropout = nn.Dropout(p=0.3)
        self.embracenet = EmbraceNet(device=device, input_size_list=[self.text_context_embedding_size,self.text_context_embedding_size,self.text_embedding_size,self.audio_embedding_size,self.au_embedding_size],embracement_size=self.embracement_size,all_att=all_att,malefic=malefic)

        # post embracement layers
        self.linear = nn.Linear(self.embracement_size, self.embracement_size // 2)
        self.classifier = nn.Linear(self.embracement_size // 2, 3)

    def eval(self):
        super().eval()
        self.is_training = False

    def train(self, bool=True):
        super().train(bool)
        self.is_training = bool

    def forward(self, text,pad_text,prosody,pad_prosody,au, pad_au, present_au,
                pretraining=False,return_embeddings=False) :

        # get embeddings for each modality
        x_au = self.au_encoder(au, pad_au)
        x_text_context_therapist,x_text_context,x_text_utterance = self.text_encoder(text, pad_text)
        x_audio, audio_padding = self.audio_encoder(prosody,pad_prosody)
	    # drop unavailable modalities
        availabilities = torch.ones([au.shape[0], self.n_modalities], device=self.device)
        availabilities[:, 0] = (torch.sum(pad_text[:,0],dim=1)>2)
        availabilities[:, 1] = (torch.sum(pad_text[:, 1], dim=1) > 2)
        availabilities[:, self.n_modalities-1] = present_au

        if (self.model_drop_text):
            availabilities[:, 0] = 0
            availabilities[:, 1] = 0
            availabilities[:, 2] = 0

        if (self.model_drop_audio):
            availabilities[:, 3] = 0
        if (self.model_drop_face):
            availabilities[:, 4] = 0


        # modality dropout during training
        if (self.is_training and self.model_dropout):
            dropout_prob = torch.rand(1, device=self.device)[0]
            if (dropout_prob >= 0.5):

                target_modalities = torch.randint(0, 2, (au.shape[0], self.n_modalities), device=self.device)
                availabilities_test = availabilities + target_modalities
                availabilities_test[availabilities_test < 2] = 0
                availabilities_test[availabilities_test == 2] = 1
                ok_rows = torch.sum(availabilities_test, dim=1)
                ok_rows[ok_rows > 0] = 1
                ok_rows = torch.cat([ok_rows.unsqueeze(dim=1) for i in range(self.n_modalities)], dim=1)
                ok_rows = 1 - ok_rows

                availabilities = availabilities_test + torch.mul(ok_rows, availabilities)

        # embrace
        
        input_list = [x_text_context_therapist,x_text_context,x_text_utterance, x_audio,x_au]
        pad_au[:, 0] = 0
        #fusion the embeddings
        x_embrace, x_weights,docking_output_stack = self.embracenet(input_list, availabilities=availabilities, return_embeddings=True,padding_list=[(1-pad_text[:,0]).type(torch.bool),(1-pad_text[:,1]).type(torch.bool),(1-pad_text[:,2]).type(torch.bool),audio_padding.type(torch.bool),pad_au])
        # employ final layers
        x_embrace = self.linear(x_embrace)
        x_embrace = self.dropout(x_embrace)
        x_embrace = nn.functional.leaky_relu(x_embrace)
        x = self.classifier(x_embrace)
        # output softmax
        if return_embeddings == True:
            return x, x_weights,docking_output_stack[:,:,0],docking_output_stack[:,:,1],docking_output_stack[:,:,2],docking_output_stack[:,:,3],docking_output_stack[:,:,4]
        return x, x_weights
