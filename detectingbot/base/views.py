import matplotlib
matplotlib.use("Agg")   # Must come before pyplot or seaborn
import matplotlib.pyplot as plt
import base64
import io
import pickle
import pandas as pd
import requests
from django import urls
from django.shortcuts import render
from django.http import HttpResponse
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import urllib
from django.views.decorators.csrf import csrf_exempt
from tweepy.errors import TooManyRequests
from bot_detection_model.train_model import TrainModel
from bot_detection_model.get_twitter_details import TwitterDetails

# Store last predictions in memory (resets on server restart)
RECENT_ACTIVITY = []
MAX_RECENT = 5  # Show last 5 predictions


def home(request):
    plt.plot(range(10))
    fig = plt.gcf()
    # convert graph into string buffer and then convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'base/index.html', {'data': uri})


def user(request):
    username = request.GET['username']
    print(username)
    return render(request, 'base/user.html', {'name': username})


def bot(request):
    return render(request, 'base/botanalysis.html')


def dashboard(request):
    train_model = TrainModel()
    
    # Total tweets analyzed (sum of statuses_count in train data)
    df_train = train_model.load_data('data.csv')
    total_tweets = df_train['statuses_count'].sum() if 'statuses_count' in df_train.columns else len(df_train)

    # Number of bots detected in recent activity
    bots_detected = sum(1 for item in RECENT_ACTIVITY if item['status'] == 'Bot')

    # Users monitored (unique usernames in recent activity)
    users_monitored = len(set(item['username'] for item in RECENT_ACTIVITY))

    context = {
        'total_tweets': total_tweets,
        'bots_detected': bots_detected,
        'accuracy': 76,  # Optional
        'users_monitored': users_monitored,
        'recent_activity': list(reversed(RECENT_ACTIVITY))  # Show latest first
    }

    return render(request, 'base/dashboard.html', context)


def train_model(request):
    train_model = TrainModel()
    train_model.train()
    return render(request, 'base/index.html')

@csrf_exempt
def prediction(request):
    train_model = TrainModel()
    context = {}

    if request.method == 'POST':
        url = request.POST.get('url')
        username = url.split('/')[-1]
        twitterapi = TwitterDetails()

        try:
            twitterapi.get_user_details(username)
            output = train_model.my_predict()[0]  # Get single prediction

            status = 'Bot' if output == 1 else 'Not Bot'

            # Store prediction in global recent activity
            confidence = "N/A"  # Optional: if you have probabilities, you can calculate
            RECENT_ACTIVITY.append({
                'username': username,
                'status': status,
                'confidence': confidence
            })

            # Keep only last MAX_RECENT items
            if len(RECENT_ACTIVITY) > MAX_RECENT:
                RECENT_ACTIVITY.pop(0)

            context = {
                'output': status
            }

        except TooManyRequests:
            context = {'output': 'Error: Too Many Requests. Please wait a few minutes.'}

    return render(request, 'base/index.html', context)
