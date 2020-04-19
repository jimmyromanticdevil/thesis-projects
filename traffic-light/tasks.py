import os
import time
from celery import Celery
from celery.task import Task
from celery.registry import tasks
#connetion string celery. utk berinteraksi dengan broker AMQP
#arg1 tasks module kt yg dimana tasks.py
#arg2 url msg broker yang akn kt gunakan dlm case ini settingan default rabbitmq amqp://localhost kl redis redis://localhost
#
from algorthms.utils import handler as algorthms_handler
celery = Celery('tasks', broker='amqp://guest@localhost//')

#AMQP mendapat(Receive) msg task dari client(django) 
#Para worker akn terus-menerus memantau queue; setelah msg di drop ke queue,maka salah satu pekerja akn mengambilnya dan mengeksekusinya.
class Traffics(Task):
      def run(self, lemeina, perintis, rate):
        poster = algorthms_handler.doit_now(lemeina, perintis, rate)
        return poster
#register class yg sudh di buat ke tasks
tasks.register(Traffics)

