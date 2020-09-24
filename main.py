from flask import Flask
from threading import Event
import json
from flask_kafka import FlaskKafka
from producer import ml_producer
from keras.models import load_model
from data_utils import Data, from_categorical, from_percent
import os

app = Flask(__name__)

INTERRUPT_EVENT = Event()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load models
model_year = load_model('./models/model_year')
model_month = load_model('./models/model_month')
model_pmnt_type_zp = load_model('./models/model_pmnt_type_zp')
model_pmnt_type_tax = load_model('./models/model_pmnt_type_tax')
model_tax_type = load_model('./models/model_tax_type')

# create producer class instance
producer = ml_producer()

# load config
config = json.load(open('config.json', encoding='utf-8'))


def predict_data(msg):
    input_data = msg
    text_data = Data(json_data=input_data['TextData'])
    input_data['PStatus'] = 1
    text_data.preprocess_data()
    # predict period (year and month)
    text_data.set_alphabet(config['data']['period_alphabet'])
    model_input = text_data.get_data()
    prediction = model_year.predict(model_input)
    input_data['PYear'], input_data['PStatus'] = from_categorical(prediction, 0.9, config['data']['init_year'], input_data['PStatus'])
    prediction = model_month.predict(model_input)
    input_data['PMonth'], input_data['PStatus'] = from_categorical(prediction, 0.9, 0, input_data['PStatus'])
    # predict types
    text_data.preprocess_type_text()
    text_data.set_alphabet(config['data']['type_alphabet'])
    model_input = text_data.get_data()
    if input_data['Tax'] == 0:  # prediction if wage
        input_data['TaxType'] = None
        prediction = model_pmnt_type_zp.predict(model_input)
        input_data['PmntType'], input_data['PStatus'] = from_percent(prediction, 0.9, input_data['PStatus'])
    else:  # prediction if tax
        prediction = model_tax_type.predict(model_input)
        input_data['TaxType'], input_data['PStatus'] = from_categorical(prediction, 0.9, 0, input_data['PStatus'])
        prediction = model_pmnt_type_tax.predict(model_input)
        input_data['PmntType'], input_data['PStatus'] = from_percent(prediction, 0.9, input_data['PStatus'])
    del input_data['TextData']
    del input_data['Tax']
    return input_data


def json_deserializer(data):
    return json.loads(data.decode('utf-8'))


bus = FlaskKafka(INTERRUPT_EVENT,
                 bootstrap_servers=",".join(["localhost:9092"]),
                 value_deserializer = json_deserializer,
                 group_id="consumer-grp-id"
                 )


@bus.handle('TO_ML')
def test_topic_handler(msg):
    print("consumed data from TO_ML topic")
    producer.send_message(predict_data(msg.value))


if __name__ == '__main__':
    bus.run()
    app.run(debug=True, port=5004)
