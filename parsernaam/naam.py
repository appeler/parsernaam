#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pkg_resources import resource_filename
import pandas as pd
import torch
import joblib

from .model import LSTM

class Parsernaam:
    """
    Parse names
    """


    @staticmethod
    def parse(df: pd.DataFrame, model_fn: str, vocab_fn: str) -> pd.DataFrame:
        """
        Parse names
        """
        MODEL = resource_filename(__name__, model_fn)
        VOCAB = resource_filename(__name__, vocab_fn)

        vectorizer = joblib.load(VOCAB)
        vocab = list(vectorizer.get_feature_names_out())
        n_letters = len(vocab)
        all_letters = ''.join(vocab)
        oob = n_letters + 1
        all_categories = ['last', 'first']
        n_categories = len(all_categories)

        n_hidden = 256
        seq_len = 30
        vocab_size = n_letters + 1 + 1 # vocab + oob + 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        model = LSTM(vocab_size, n_hidden, n_categories, num_layers=2)
        model.load_state_dict(torch.load(MODEL, map_location=device))
        model.to(device)

        # set the model to evaluation mode
        model.eval()

        def letterToIndex(letter):
            return all_letters.find(letter)

        def lineToTensor(line):
            tensor = torch.ones(seq_len) * oob
            try:
                for li, letter in enumerate(line):
                    tensor[li] = letterToIndex(letter)
            except:
                pass
            return tensor

        def name_parser(name):
            name_types = []
            names = name.split()
            for n in names:
                name_tokens = lineToTensor(n)
                out = model(name_tokens.unsqueeze(0).to(device))
                probs = torch.exp(out)
                out = torch.argmax(probs)
                name_type = all_categories[out.item()]
                name_types.append({'name': n, 'type': name_type, 'prob': probs[0][out].item()})
            return name_types

        df['parsed_name'] = df['name'].apply(name_parser)
        return df
