import torch
import torch.nn as nn


class EmbraceNet(nn.Module):
  
  def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False,all_att=False,malefic=True):
    """
    Initialize an EmbraceNet module.
    Args:
      device: A "torch.device()" object to allocate internal parameters of the EmbraceNet module.
      input_size_list: A list of input sizes.
      embracement_size: The length of the output of the embracement layer ("c" in the paper).
      bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
    """
    super(EmbraceNet, self).__init__()

    self.device = device
    self.input_size_list = input_size_list
    self.random_indice_selection = all_att
    self.embracement_size = embracement_size
    self.dropout=nn.Dropout(p=0.3)
    self.bypass_docking = bypass_docking
    self.malefic=malefic

    self.cross_attn_text = nn.MultiheadAttention(embracement_size, 1)
    self.cross_attn_text_context = nn.MultiheadAttention(embracement_size, 1)
    self.cross_attn_audio = nn.MultiheadAttention(embracement_size, 1)
    self.cross_attn_face = nn.MultiheadAttention(embracement_size, 1)
    self.cross_attn_text_context_therapist = nn.MultiheadAttention(embracement_size, 1)



    if (not bypass_docking):
      for i, input_size in enumerate(input_size_list):
        setattr(self, 'docking_%d' % (i), nn.Sequential(self.dropout,nn.LeakyReLU(),nn.Linear(input_size, embracement_size)))
        setattr(self, 'docking2_%d' % (i), nn.Sequential(self.dropout,nn.LeakyReLU(),nn.Linear(input_size,embracement_size)))
        #setattr(self, 'docking3_%d' % (i), nn.Sequential(self.dropout,nn.LeakyReLU(),nn.Linear(embracement_size,embracement_size)))



  def forward(self, input_list, availabilities=None, selection_probabilities=None,return_embeddings=False,padding_list = None,eval=False):
    """
    Forward input data to the EmbraceNet module.
    Args:
      input_list: A list of input data. Each input data should have a size as in input_size_list.
      availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
      selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
    Returns:
      A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
    """

    # check input data
    assert len(input_list) == len(self.input_size_list)
    num_modalities = len(input_list)
    batch_size = input_list[0].shape[0]
    

    # docking layer
    docking_output_list = []
    docking_output_list_stack = []
    if (self.bypass_docking):
      docking_output_list = input_list
    else:
      for i, input_data in enumerate(input_list):
        x = getattr(self, 'docking_%d' % (i))(input_data)
        x2 = input_data.clone()

        if len(x.shape)< 3:
          x = x.unsqueeze(1)
        docking_output_list.append(x)

        x2[padding_list[i].unsqueeze(-1).expand_as(x2).to(bool)] = 0
        x2 = x2.sum(dim=1)

        x2 = x2 / (
                (~padding_list[i].to(bool)).sum(dim=1).unsqueeze(-1).expand_as(x2) + 0.00001)

        #x2 = getattr(self, 'docking3_%d' % (i))(x2)
        x2 = getattr(self, 'docking2_%d' % (i))(x2)
        docking_output_list_stack.append(x2)
    

    # check availabilities
    if (availabilities is None):
      availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
    else:
      availabilities = availabilities.float()
    

    # stack docking outputs
    docking_output_stack = torch.stack(docking_output_list_stack, dim=-1)  # [batch_size, embracement_size, num_modalities]


    # embrace
   # modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
    #modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]


    x_weights_text = self.cross_attn_text(docking_output_list[0].permute(1,0,2), docking_output_list[0].permute(1,0,2), docking_output_list[0].permute(1,0,2),key_padding_mask=padding_list[0])[0].permute(1,0,2)
    x_weights_text_context = \
    self.cross_attn_text_context(docking_output_list[1].permute(1, 0, 2), docking_output_list[1].permute(1, 0, 2),
                         docking_output_list[1].permute(1, 0, 2), key_padding_mask=padding_list[1])[0].permute(1, 0, 2)
    x_weights_text_context_therapist = \
      self.cross_attn_text_context_therapist(docking_output_list[2].permute(1, 0, 2), docking_output_list[2].permute(1, 0, 2),
                                   docking_output_list[2].permute(1, 0, 2), key_padding_mask=padding_list[2])[
        0].permute(1, 0, 2)
    x_weights_audio = \
        self.cross_attn_audio(docking_output_list[3].permute(1, 0, 2), docking_output_list[3].permute(1, 0, 2),
                                docking_output_list[3].permute(1, 0, 2), key_padding_mask=padding_list[3])[0].permute(1, 0, 2)
    x_weights_face = \
        self.cross_attn_face(docking_output_list[4].permute(1, 0, 2), docking_output_list[4].permute(1, 0, 2),
                                docking_output_list[4].permute(1, 0, 2), key_padding_mask=padding_list[4])[0].permute(1, 0, 2)

    x_weights_text = self.dropout(x_weights_text)
    x_weights_text_context = self.dropout(x_weights_text_context)
    x_weights_text_context_therapist = self.dropout(x_weights_text_context_therapist)
    x_weights_audio = self.dropout(x_weights_audio)
    x_weights_face = self.dropout(x_weights_face)
    x_weights_text[padding_list[0].unsqueeze(-1).expand_as(x_weights_text).to(bool)] = 0

    x_weights_text = x_weights_text.sum(dim=1)

    x_weights_text = x_weights_text / (
            (~padding_list[0].to(bool)).sum(dim=1).unsqueeze(-1).expand_as(x_weights_text) + 0.00001)

    x_weights_text_context[padding_list[1].unsqueeze(-1).expand_as(x_weights_text_context).to(bool)] = 0
    x_weights_text_context = x_weights_text_context.sum(dim=1)

    x_weights_text_context = x_weights_text_context / (
            (~padding_list[1].to(bool)).sum(dim=1).unsqueeze(-1).expand_as(x_weights_text) + 0.00001)

    x_weights_text_context_therapist[padding_list[2].unsqueeze(-1).expand_as(x_weights_text_context_therapist).to(bool)] = 0
    x_weights_text_context_therapist = x_weights_text_context_therapist.sum(dim=1)

    x_weights_text_context_therapist = x_weights_text_context_therapist / (
            (~padding_list[2].to(bool)).sum(dim=1).unsqueeze(-1).expand_as(x_weights_text) + 0.00001)

    x_weights_audio[padding_list[3].unsqueeze(-1).expand_as(x_weights_audio).to(bool)] = 0
    x_weights_audio = x_weights_audio.sum(dim=1)

    x_weights_audio = x_weights_audio / (
            (~padding_list[3].to(bool)).sum(dim=1).unsqueeze(-1).expand_as(x_weights_text) + 0.00001)

    x_weights_face[padding_list[4].unsqueeze(-1).expand_as(x_weights_face).to(bool)] = 0
    x_weights_face = x_weights_face.sum(dim=1)

    x_weights_face = x_weights_face / (
            (~padding_list[4].to(bool)).sum(dim=1).unsqueeze(-1).expand_as(x_weights_text) + 0.00001)
    if self.random_indice_selection :

      x_weights_text = torch.mean(x_weights_text, dim=1)
      x_weights_text_context= torch.mean(x_weights_text_context, dim=1)
      x_weights_text_context_therapist = torch.mean(x_weights_text_context_therapist, dim=1)
      x_weights_audio = torch.mean(x_weights_audio, dim=1)
      x_weights_face = torch.mean(x_weights_face, dim=1)
      x_weights = torch.stack((x_weights_text,x_weights_text_context,x_weights_text_context_therapist,x_weights_audio,x_weights_face),dim=1)

      availabilities_inf = torch.mul(-100000 * torch.ones(availabilities.shape, device=self.device), 1 - availabilities)
      x_weights = nn.functional.sigmoid(x_weights) +0.000001
      x_weights = torch.mul(x_weights, availabilities) #+ availabilities_inf
      x_weights = x_weights/ (torch.sum(x_weights, dim=1).unsqueeze(-1).expand_as(x_weights))

      modality_indices = torch.multinomial(x_weights, num_samples=self.embracement_size,
                                         replacement=True)  # [batch_size, embracement_size]
      #modality_indices = torch.stack([torch.multinomial(x_weights[:,:,i],num_samples=1) for i in range(self.embracement_size)],dim=1).squeeze()
      modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]
    else:

      x_weights = torch.stack((x_weights_text,x_weights_text_context,x_weights_text_context_therapist,x_weights_audio,x_weights_face),dim=1)
      #x_weights = nn.functional.sigmoid(x_weights) + 0.000001

      x_weights = x_weights+ -10000000*(1-availabilities.unsqueeze(-1).expand_as(x_weights))
      x_weights = nn.functional.softmax(x_weights,dim=1)
      x_weights = torch.mul(x_weights, availabilities.unsqueeze(-1).expand_as(x_weights))
      if not self.malefic:
        x_weights = 0*x_weights + 1
      if self.eval and self.malefic:

        modality_indices = torch.argmax(x_weights, dim=1)  # [batch_size, embracement_size]
      else:
        try:
          modality_indices = torch.stack(
          [torch.multinomial(x_weights[:, :,i], num_samples=1) for i in range(self.embracement_size)], dim=1).squeeze()
        except:
          for i in range(self.embracement_size):
            print(x_weights[:, :, i])
            torch.multinomial(x_weights[:, :, i], num_samples=1)
          raise
      modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()
    
    embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
    embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]
    if return_embeddings:
      return embracement_output,x_weights, docking_output_stack
    return embracement_output,x_weights
