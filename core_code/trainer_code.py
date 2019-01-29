import sys, os

current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(current_file_path + '/../../')
sys.path.append(root_dir)  # To get the Root the directory

from core_code.Feature_Extraction.AsthmaAnalyzer import AsthmaAnalyzer
from core_code.ML.svmmodel import SVMModel
import ast
import numpy as np
import json
from collections import namedtuple
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder


def get_datasets():
    db = connect_db()
    all_records = db.records.find({})
    labels = []
    breath_features = []
    for single_record in all_records:
        single_label = 1 if single_record.get('is_urgent_label') == "1" else -1
        labels.append(single_label)
        # print(single_label)
        single_breath_features = single_record.get('features')
        for feature_name, feature_value in single_breath_features.items():
            try:
                single_breath_features[feature_name] = np.array(ast.literal_eval(feature_value))
            except:
                single_breath_features[feature_name] = np.array(ast.literal_eval(feature_value.replace("NaN", "0")))
        breath_features.append(single_breath_features)
    return breath_features, labels


def connect_db():
    # 2: Load the env file
    dotenv_path = root_dir + '/config/.env'
    load_dotenv(dotenv_path)

    client = pymongo.MongoClient('localhost', 27017)
    db = client[os.getenv('DB_DATABASE')]
    return db
    # 1: Load the SSH file
    ssh_path = root_dir + '/config/ssh_credentials.json'
    credentials = ''
    with open(ssh_path) as f:
        credentials = json.load(f)

    ssh_username = credentials["user"]
    ssh_passphrase = credentials["pass_phrase"]
    ssh_priv_key = credentials["private_key"]
    ssh_host = credentials["host"]

    server = SSHTunnelForwarder(
        ssh_host,
        ssh_username=ssh_username,
        ssh_private_key=ssh_priv_key,
        ssh_private_key_password=ssh_passphrase,
        remote_bind_address=('localhost', int(os.getenv('DB_PORT')))
    )

    server.start()

    client = pymongo.MongoClient('127.0.0.1', server.local_bind_port)
    db = client[os.getenv('DB_DATABASE')]
    return db


def train_svm(features, labels):
    svm_model = SVMModel(features, labels)
    svm_model.train_the_system()
    pass


def train_knn(features, labels):
    pass


def train_ann(features, labels):
    pass


start = time.time()
print("\n Start Time  = " + str(start) + "\n")
features, labels = get_datasets()
End = time.time()
elapsed = End - start
print("End Time = " + str(End) + "\n")
print("Difference = " + str(elapsed))

train_svm(features, labels)
train_knn(features, labels)
train_ann(features, labels)


