from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('conversations/', views.conversations_list, name='conversations_list'),
    path('conversations/<str:conversation_id>/', views.conversation_detail, name='conversation_detail'),
    path('alphabet/', views.alphabet, name='alphabet'),
    path('vocabulary/', views.vocabulary, name='vocabulary'),
    path('phrases/', views.phrases, name='phrases'),
    path('numbers/', views.numbers, name='numbers'),
]
