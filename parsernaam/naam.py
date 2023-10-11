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
    def parse(df: pd.DataFrame, model_fn: str, model_fn_pos: str, vocab_fn: str) -> pd.DataFrame:
        """
        Parse names
        """
        MODEL = resource_filename(__name__, model_fn)
        MODEL_POS = resource_filename(__name__, model_fn_pos)
        VOCAB = resource_filename(__name__, vocab_fn)

        vectorizer = joblib.load(VOCAB)
        vocab = list(vectorizer.get_feature_names_out())
        n_letters = len(vocab)
        all_letters = ''.join(vocab)
        oob = n_letters + 1

        all_categories_pos = ['last_first', 'first_last']
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

        model_pos = LSTM(vocab_size, n_hidden, len(all_categories_pos), num_layers=2)
        model_pos.load_state_dict(torch.load(MODEL_POS, map_location=device))
        model_pos.to(device)

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
            names = name.split()
            name_tokens = lineToTensor(name)
            # if there is only one name, use the non-pos model
            # otherwise use the pos model
            if len(names) == 1:
                out = model(name_tokens.unsqueeze(0).to(device))
                probs = torch.exp(out)
                out = torch.argmax(probs)
                name_type = all_categories[out.item()]
            else:
                out = model_pos(name_tokens.unsqueeze(0).to(device))
                probs = torch.exp(out)
                out = torch.argmax(probs)
                name_type = all_categories_pos[out.item()]
            return {'name': name, 'type': name_type, 'prob': probs[0][out].item()}

        df['parsed_name'] = df['name'].apply(name_parser)
        return df
