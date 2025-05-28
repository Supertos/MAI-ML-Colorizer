from celery import Celery


celery = Celery(
    "worker",
    broker="amqp://guest:guest@rabbitmq:5672//",
    backend="rpc://"
)

import app.tasks