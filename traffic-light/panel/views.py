# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.template import Context, RequestContext,Template
from django.core.urlresolvers import reverse
from django.utils.encoding import smart_str, smart_unicode
from django.core.cache import cache
from django.views.generic.simple import direct_to_template
from django.shortcuts import get_object_or_404
from django import http
from django.core import urlresolvers
from django import template
import time
from time import localtime
from datetime import datetime, timedelta
import sys
import random
import os
import hashlib
import ast
#mengimport method yang ada di tasks.py
#yang akan kt gunakan utk mendrop msg ke queue
from tasks import Traffics

def task(request):
    if request.POST:
       try:
          at_leimena = request.POST.get('at_leimena')
          at_printis = request.POST.get('at_printis')
          at_rate = request.POST.get('at_rate')
          at_leimena = ast.literal_eval(at_leimena)
          at_printis = ast.literal_eval(at_printis)
          at_rate = ast.literal_eval(at_rate)
          result_one = Traffics.delay(lemeina=at_leimena, perintis=at_printis, rate=at_rate) 
       except Exception,e:
          print e
          return HttpResponse('Mohon isi data dengan valid')
       return HttpResponseRedirect('/')
    return render_to_response('forms.html',locals(),context_instance=RequestContext(request))            

