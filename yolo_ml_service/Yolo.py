import errno

import logging

import sys

import torch


class Yolo:
    def __init__(self):
        self.loaded = False
        self.model_name = "Yolo"
        self.model_file = "../models/yolov7.pt"

    def load(self):
        logging.info(f"Loading model from {self.model_file}")

        try:
            self.model = torch.load(self.model_file, map_location=torch.device('cpu'))
        except IOError:
            logging.exception(f"Unable to load the modelfile: {self.model_file}")
            sys.exit(errno.ENOENT)

        self.model.eval()

    def predict(self, X, features_names):
        logging.debug(f"Performing prediction on feature_names: {features_names} and X: {X}")

        if not self.loaded:
            self.load()

        # perform prediction here
            
        return X
