import matplotlib
matplotlib.use("Agg")   # Must come before pyplot or seaborn
import matplotlib.pyplot as plt
import base64
import io
import requests
from django import urls
from django.shortcuts import render
from django.http import HttpResponse
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import urllib
from bot_detection_model.train_model import TrainModel
from bot_detection_model.get_twitter_details import TwitterDetails


def home(request):
    plt.plot(range(10))
    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'base/index.html', {'data': uri})


def about(request):
    return render(request, 'base/about.html')


def user(request):
    username = request.GET['username']
    print(username)
    return render(request, 'base/user.html', {'name': username})


def bot(request):
    return render(request, 'base/botanalysis.html')

def train_model(request):
    train_model = TrainModel()
    train_model.train()
    return render(request, 'base/index.html')

def prediction(request):
    train_model = TrainModel()
    if request.method == 'POST':
        url = request.POST.get('url')
    username = url.split('/')[-1]
    twitterapi = TwitterDetails()
    twitterapi.get_user_details(username)
    output=train_model.my_predict()
    if output==0:
        context={'output':'Not Bot'}
    else:
        context={'output':'Bot'}
    return render(request, 'base/index.html',context)
