from kafka import KafkaProducer
from serializers import json_serializer


class ml_producer(object):
    def __init__(self, bootstrap_servers=['localhost:9092'],
                 topic_name='FROM_ML'):
        self.topic_name = topic_name
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=json_serializer
        )

    def on_send_success(self, record_metadata):
        print('Produced data to {0} topic, partition:{1}, offset:{2}'.format(record_metadata.topic, record_metadata.partition, record_metadata.offset))

    def on_send_error(self, excp):
        print(excp)

    def send_message(self, msg):
        self.producer.send(self.topic_name, msg).add_callback(self.on_send_success).add_errback(self.on_send_error)
        self.producer.flush()
