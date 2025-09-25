from unicodedata import name
from django.contrib import admin
from django.urls import path
from . import views

app_name= 'base'
urlpatterns = [
    path('', views.home, name="home"),
    path('about/', views.about, name="about"),
    path('user', views.user, name="user"),
    path('bot/', views.bot, name="bot"),
    path('train/', views.train_model, name="train_model"),
    path('prediction', views.prediction, name="prediction"),
    ]

