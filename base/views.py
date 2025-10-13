import matplotlib
matplotlib.use("Agg")  # Must come before pyplot or seaborn
import matplotlib.pyplot as plt
import base64
import io
import pickle
import pandas as pd
import urllib
from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tweepy.errors import TooManyRequests
from bot_detection_model.train_model import TrainModel
from bot_detection_model.get_twitter_details import TwitterDetails

# -------------------------------
# Globals
# -------------------------------
RECENT_ACTIVITY = []
MAX_RECENT = 5  # Show last 5 predictions

# -------------------------------
# Load pre-trained model once
# -------------------------------
MODEL_PATH = Path(settings.BASE_DIR) / 'bot_detection_model' / 'outputs' / 'model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        saved_model = pickle.load(f)
    MODEL_LOADED = True
    print("✅ Pre-trained model loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load model.pkl: {e}")
    saved_model = None
    MODEL_LOADED = False


# -------------------------------
# Views
# -------------------------------

def home(request):
    """Renders the homepage with a simple example plot."""
    plt.plot(range(10))
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'base/index.html', {'data': uri})


def user(request):
    """Render user page."""
    username = request.GET.get('username', '')
    return render(request, 'base/user.html', {'name': username})


def bot(request):
    """Render bot analysis page."""
    return render(request, 'base/botanalysis.html')


def dashboard(request):
    """Dashboard showing basic analytics."""
    train_model = TrainModel()

    # Load training data for stats
    df_train = train_model.load_data('data.csv')
    total_tweets = df_train['statuses_count'].sum() if 'statuses_count' in df_train.columns else len(df_train)

    bots_detected = sum(1 for item in RECENT_ACTIVITY if item['status'] == 'Bot')
    users_monitored = len(set(item['username'] for item in RECENT_ACTIVITY))

    context = {
        'total_tweets': total_tweets,
        'bots_detected': bots_detected,
        'accuracy': 76,  # Example static accuracy metric
        'users_monitored': users_monitored,
        'recent_activity': list(reversed(RECENT_ACTIVITY)),  # Latest first
    }
    return render(request, 'base/dashboard.html', context)


def train_model_view(request):
    """
    Optional manual re-training view.
    Disabled automatically on Render (DEBUG=False).
    """
    if settings.DEBUG:
        tm = TrainModel()
        tm.train()
        message = "✅ Model retrained successfully (local)."
    else:
        message = "⚠️ Training disabled in production (Render). Using pre-trained model."

    return render(request, 'base/index.html', {'message': message})


@csrf_exempt
def prediction(request):
    """Predicts whether a given Twitter user is a bot or not."""
    context = {}

    if request.method == 'POST':
        url = request.POST.get('url', '')
        username = url.split('/')[-1] if url else ''
        twitterapi = TwitterDetails()

        try:
            twitterapi.get_user_details(username)

            if MODEL_LOADED and saved_model:
                # Use pre-trained model for prediction
                tm = TrainModel()
                X_test = tm.make_predict_data()
                output = saved_model.predict(X_test)[0]
                status = 'Bot' if output == 1 else 'Not Bot'
            else:
                status = "Model not available"

            # Store prediction in memory
            RECENT_ACTIVITY.append({
                'username': username,
                'status': status,
                'confidence': "N/A"
            })
            if len(RECENT_ACTIVITY) > MAX_RECENT:
                RECENT_ACTIVITY.pop(0)

            context = {'output': status}

        except TooManyRequests:
            context = {'output': 'Error: Too Many Requests. Please wait a few minutes.'}
        except Exception as e:
            context = {'output': f'Error: {str(e)}'}

    return render(request, 'base/index.html', context)
