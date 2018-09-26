import os
from pandas.io.json import json_normalize
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import ujson
import cloudpickle as pickle
import logging

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
    'model_name': 'safedriver',
    'model_tag': 'v1',
    'model_type': 'scikit',
    'model_runtime': 'python',
    'model_chip': 'cpu',
}


def _initialize_upon_import():
    model_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

    # Load pickled model from model directory
    with open(model_pkl_path, 'rb') as fh:
        restored_model = pickle.load(fh)

    return restored_model


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""
    transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.predict(transformed_request)

    return _transform_response(response)


@monitor(labels=_labels, name="transform_request")
def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = ujson.loads(request_str)
    return json_normalize(request_json)


@monitor(labels=_labels, name="transform_response")
def _transform_response(response):
    return ujson.dumps(response.tolist())


if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)
