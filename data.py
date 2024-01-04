import os
import numpy as np
from utils import clean_text, label_to_target
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer
def prepare_text_datatset(df, features_text):
    df = pd.DataFrame(df, columns=features_text)
    for t in features_text:
        df[t + '_clean'] = df[t].apply(lambda x: clean_text(x))

    features_text_clean = [t + '_clean' for t in features_text]
    df = df[features_text_clean]
    return df

def create_dataset(dataset_path):

    features_text = ["previous context", "context","text"]
    model_name='bert-base-uncased'

    prosody_train = np.load(dataset_path + "/prosody_train.npy", allow_pickle=True)
    prosody_test = np.load(dataset_path + "/prosody_val.npy", allow_pickle=True)
    pad_prosody_train = np.load(dataset_path + "/pad_prosody_train.npy", allow_pickle=True)
    pad_prosody_test = np.load(dataset_path + "/pad_prosody_val.npy", allow_pickle=True)
    y_train = np.load(dataset_path + "/y_train.npy", allow_pickle=True)
    y_test = np.load(dataset_path + "/y_val.npy", allow_pickle=True)
    au_train = np.load(dataset_path + "/au_train.npy", allow_pickle=True)
    au_test = np.load(dataset_path + "/au_val.npy", allow_pickle=True)
    present_face_test = np.load(dataset_path + "/present_face_val.npy", allow_pickle=True)
    present_face_train = np.load(dataset_path + "/present_face_train.npy", allow_pickle=True)
    pad_au_train = np.load(dataset_path + "/pad_au_train.npy", allow_pickle=True)
    pad_au_test = np.load(dataset_path + "/pad_au_val.npy", allow_pickle=True)
    text_train = np.load(dataset_path + "/text_train.npy", allow_pickle=True)
    text_test = np.load(dataset_path + "/text_val.npy", allow_pickle=True)
    text_train = prepare_text_datatset(text_train, features_text)
    text_test = prepare_text_datatset(text_test, features_text)

    present_face_train = torch.Tensor(present_face_train)
    present_face_test = torch.Tensor(present_face_test)
    pad_prosody_train = torch.Tensor(pad_prosody_train)
    pad_prosody_test = torch.Tensor(pad_prosody_test)

    y_train = torch.Tensor([label_to_target[i] for i in y_train])
    y_test = torch.Tensor([label_to_target[i] for i in y_test])
    pad_au_train = torch.Tensor(pad_au_train)
    pad_au_test = torch.Tensor(pad_au_test)

    inputs_train = np.array(text_train)
    inputs_test = np.array(text_test)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

    max_len = 0

    # For every sentence...
    for sent in inputs_train:
        input_ids = []
        for i in range(len(features_text)):
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids += [tokenizer.encode(sent[i], add_special_tokens=True)]

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids[0]))

    for sent in inputs_test:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        for i in range(len(features_text)):
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids += [tokenizer.encode(sent[i], add_special_tokens=True)]

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids[0]))
    input_ids_train = []
    attention_masks_train = []

    # For every tweet...
    for sent in inputs_train:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = []

        for i in range(len(features_text)):
            encoded_dict += [tokenizer.encode_plus(
                sent[i],  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )]

        if len(features_text) > 0:
            # Add the encoded sentence to the list.
            input_ids_train.append(
                torch.stack([encoded_dict[i]['input_ids'] for i in range(len(features_text))], dim=1))

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks_train.append(
                torch.stack([encoded_dict[i]['attention_mask'] for i in range(len(features_text))], dim=1))

    input_ids_test = []
    attention_masks_test = []
    for sent in inputs_test:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = []
        for i in range(len(features_text)):
            encoded_dict += [tokenizer.encode_plus(
                sent[i],  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )]
        if len(features_text) > 0:
            # Add the encoded sentence to the list.
            input_ids_test.append(torch.stack([encoded_dict[i]['input_ids'] for i in range(len(features_text))], dim=1))

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks_test.append(
                torch.stack([encoded_dict[i]['attention_mask'] for i in range(len(features_text))], dim=1))
    # Convert the lists into tensors.
    if len(features_text) > 0:
        input_ids_train = torch.cat(input_ids_train, dim=0)
        attention_masks_train = torch.cat(attention_masks_train, dim=0)
        input_ids_test = torch.cat(input_ids_test, dim=0)
        attention_masks_test = torch.cat(attention_masks_test, dim=0)


    prosody_train = torch.tensor(prosody_train)
    prosody_test = torch.tensor(prosody_test)
    openface_train = torch.tensor(au_train)
    openface_test = torch.tensor(au_test)

    target = y_train

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count


    samples_weight = np.array([weight[int(t)] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_dataset = TensorDataset(prosody_train,pad_prosody_train,y_train,input_ids_train,attention_masks_train,openface_train,pad_au_train,present_face_train)
    val_dataset = TensorDataset(prosody_test,pad_prosody_test,y_test,input_ids_test,attention_masks_test,openface_test,pad_au_test,present_face_test)
    batch_size = 4

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=sampler,  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        shuffle = False,  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader
