import nltk
from nltk.corpus import stopwords
import re
import numpy as np

sw = stopwords.words('english')
import datetime

label_to_target = {"sustain": 0, "neutral": 1, "change": 2}

def from_str_to_array(s):
    l = s.split(',')
    for i in range(len(l)-1):
        l[i] = float(l[i][1:])
    l[-1] = float(l[-1][1:-1])
    return l

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ",
                  text)  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^,' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')  # Removing punctuations

    text = [word.lower() for word in text.split()]

    text = " ".join(text)  # lower text

    return text

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#%%
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
