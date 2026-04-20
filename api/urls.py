from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('alphabet/', views.alphabet, name='alphabet'),
    path('vocabulary/', views.vocabulary, name='vocabulary'),
    path('phrases/', views.phrases, name='phrases'),
    path('numbers/', views.numbers, name='numbers'),
]
