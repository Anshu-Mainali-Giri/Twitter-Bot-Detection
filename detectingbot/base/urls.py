from django.contrib import admin
from django.urls import path
from . import views

app_name = 'base'

urlpatterns = [
    path('', views.home, name="home"),
    path('user', views.user, name="user"),
    path('bot/', views.bot, name="bot"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('train/', views.train_model, name="train_model"),
    path('prediction', views.prediction, name="prediction"),
]
