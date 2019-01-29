import numpy as np
import sys, os

current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(current_file_path + '/../../')
sys.path.append(root_dir)  # To get the Root the directory

from core_code.Feature_Extraction.AsthmaAnalyzer import AsthmaAnalyzer
import json
from collections import namedtuple
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import _thread
from grpc.beta import implementations
import tensorflow as tf

from serving.tensorflow_serving.apis import predict_pb2
from serving.tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')


def set_up_db():
    global client
    dotenv_path = root_dir + '/config/.env'
    # Load file from the path.
    load_dotenv(dotenv_path)
    client = MongoClient("mongodb://" + os.getenv('DB_USERNAME') + ":" + os.getenv('DB_PASSWORD') + "@" + os.getenv(
        'DB_HOST') + ":" + os.getenv('DB_PORT') + "/")
    return client[os.getenv('DB_DATABASE')]


def get_breath_features_cb(audio_file_name, audio_file_path):
    asthma_analyzer = AsthmaAnalyzer('%s' % audio_file_path,
                                     '%s' % audio_file_name)
    power, energy, phase, ASE, Ti, number_of_rows, selected_data = asthma_analyzer.get_signal_features()
    return ASE, Ti, energy, number_of_rows, phase, power, selected_data


def init():
    start = time.time()
    # print("\n Start Time  = " + str(start) + "\n")
    # Steps
    # FE of the sample
    # data = "{\"label\":\"normal\",\"data_type\":\"training\",\"breath_files\":[{\"file_name\":\"normal-training-audio[0]-1526170690677.wav\",\"file_path\":\"/Users/JasemAl-sadi/WebstormProjects/asthma/samples/normal-training-audio[0]-1526170690677.wav\"},{\"file_name\":\"normal-training-audio[1]-1526170690703.wav\",\"file_path\":\"/Users/JasemAl-sadi/WebstormProjects/asthma/samples/normal-training-audio[1]-1526170690703.wav\"},{\"file_name\":\"normal-training-audio[2]-1526170690709.wav\",\"file_path\":\"/Users/JasemAl-sadi/WebstormProjects/asthma/samples/normal-training-audio[2]-1526170690709.wav\"}]}"
    if len(sys.argv) < 2:
        raise Exception('No argument sent')
    data = sys.argv[1]
    data = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    if not hasattr(data, 'breath_file'):
        raise Exception('No Breath sound file path sent, it should be as  with name breath_file')
    breath_record = data.breath_file
    if not hasattr(breath_record, 'file_name') or not hasattr(breath_record, 'file_path'):
        raise Exception('The breath file  has not  file_name or file_path attribute {} '.format(breath_record))
    audio_file_name = breath_record.file_name
    audio_file_path = breath_record.file_path
    ASE, Ti, energy, number_of_rows, phase, power, selected_data = get_breath_features_cb(audio_file_name,
                                                                                          audio_file_path)
    data_type = data.data_type
    is_urgent_label = -1
    is_urgent = 0
    # Save to Mongo db in separate thread

    try:
        _thread.start_new_thread(save_to_db, (
            ASE, Ti, audio_file_path, data_type, energy, is_urgent, is_urgent_label, phase, power))
    except:
        raise Exception("Error: unable to start thread")

    # Model prediction
    is_urgent = get_model_predictions(ASE, Ti)
    # Return the result to the node js normal=0/wheeze=1
    End = time.time()
    elapsed = End - start
    data_to_be_sent = {
        'start_time': start,
        'end_time': End,
        'difference_time': elapsed,
        'is_urgent': is_urgent,
    }
    data_to_be_sent_json = json.dumps(data_to_be_sent)

    print(data_to_be_sent_json)

    pass


def save_to_db(ASE, Ti, audio_file_path, data_type, energy, is_urgent, is_urgent_label, phase, power):
    current_record = {
        'is_urgent': is_urgent,
        'is_urgent_label': is_urgent_label,
        'type_of_record': data_type,
        'wav_file_path': audio_file_path,
        'features': {
            'ASE': json.dumps(ASE.tolist()),
            'Ti': json.dumps(Ti.tolist()),
            'Energy': json.dumps(energy.tolist()),
            'Phase': json.dumps(phase.tolist()),
            'Power': json.dumps(power.tolist()),
        },
        'created_at': time.time(),
    }
    db = set_up_db()
    db.records.insert_one(current_record).inserted_id


def get_model_predictions(ASE, Ti):
    result = 0
    data = normalize_data(np.concatenate((ASE.reshape(-1, 1), Ti.reshape(-1, 1)), axis=0)).shape(1, 960)
    data = np.concatenate([data, np.random.uniform(low=-1.0, high=1, size=(1, 960))])
    FLAGS = tf.app.flags.FLAGS
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'asthma_model'
    request.model_spec.signature_name = 'predict_asthma_attack'
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1, 960]))
    result = stub.Predict(request, 5.0)  # 10 secs timeout

    return result


def normalize_data(self, data):
    min = data.min()
    max = data.max()
    normalized_data = np.array([2 * (x - min) / (max - min) - 1 for i, x in enumerate(data)]).reshape(-1, )
    return normalized_data


init()
