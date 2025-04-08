# chatbot/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='chat_index'),
    path('chat/', views.chat_api, name='chat_api'),
]
