# -*- coding: utf-8 -*-
""" CLI for prediction

source activate condaenv
which python

python  cli_predict.py  --file_input  data/address_matching_data.csv  --folder_model models/model_01/  --folder_output  data/    > data/log_test.txt 2>&1


 > data/log_test.txt 2>&1   Re-direction



 
"""
import argparse
import logging
import os
import sys
from time import sleep

import numpy as np
import pandas as pd

####################################################################################################
import util
import util_feature
import util_model

############### Variable definition ################################################################
CWD_FOLDER = os.getcwd()


####################################################################################################
logger = logging.basicConfig()


def log(*argv):
    logger.info(",".join([str(x) for x in argv]))


####################################################################################################
def load_arguments():
    parser = argparse.ArgumentParser(description="Prediction CLI")
    parser.add_argument("--verbose", default=0, help="verbose")
    parser.add_argument("--log_file", type=str, default="log.txt", help="log file")
    parser.add_argument("--file_input", type=str, default="", help="Input file")
    parser.add_argument("--folder_output", default="nodaemon", help="Folder output")
    parser.add_argument("--folder_model", default="model/", help="Model")
    args = parser.parse_args()
    return args


def data_load(file_input):
    df = pd.read_csv(file_input)
    return df


###################################################################################
if __name__ == "__main__":
    args = load_arguments()
    logger = util.logger_setup(
        __name__, log_file=args.log_file, formatter=util.FORMATTER_4, isrotate=True
    )
    CWD_FOLDER = os.getcwd()
    log(CWD_FOLDER)

    folder_output = args.folder_output
    folder_model = args.folder_model
    file_input = args.file_input

    #############################################################################
    log("start prediction")
    log("Param load check")
    colid = "id"
    colnum = [
        "name_levenshtein_simple",
        "name_trigram_simple",
        "name_levenshtein_term",
        "name_trigram_term",
        "city_levenshtein_simple",
        "city_trigram_simple",
        "city_levenshtein_term",
        "city_trigram_term",
        "zip_levenshtein_simple",
        "zip_trigram_simple",
        "zip_levenshtein_term",
        "zip_trigram_term",
        "street_levenshtein_simple",
        "street_trigram_simple",
        "street_levenshtein_term",
        "street_trigram_term",
        "website_levenshtein_simple",
        "website_trigram_simple",
        "website_levenshtein_term",
        "website_trigram_term",
        "phone_levenshtein",
        "phone_trigram",
        "fax_levenshtein",
        "fax_trigram",
        "street_number_levenshtein",
        "street_number_trigram",
    ]

    colcat = ["phone_equality", "fax_equality", "street_number_equality"]
    coltext = []
    coldate = []
    coly = "is_match"

    #############################################################################
    log("Data load")
    df = data_load(file_input)
    df.set_index(colid)
    log("Data size", df.shape)

    #############################################################################
    log("Data preprocess")
    pipe_preprocess_colnum = util.load(folder_model + "pipe_preprocess_colnum.pkl")
    dfnum = util_feature.pd_pipeline_apply(df[colnum], pipe_preprocess_colnum)

    pipe_preprocess_colcat = util.load(folder_model + "pipe_preprocess_colcat.pkl")
    dfcat = util_feature.pd_pipeline_apply(df[colcat], pipe_preprocess_colcat)

    dfmerge = pd.concat((dfnum, dfcat), axis=1)

    #############################################################################
    log("Model predict")
    clf = util.load(folder_model + "clf_predict.pkl")
    df[coly] = clf.predict(dfmerge.values)
    log("Predict", "NUll values count", len(df[df[coly].isnull()]))

    #############################################################################
    log("Predict", "save")
    df = df.reset_index()
    dfe = df[[colid, coly]]
    dfe.to_csv(folder_output + "prediction.csv", index=False)
    log(folder_output + "prediction.csv")

    log("Finish")
