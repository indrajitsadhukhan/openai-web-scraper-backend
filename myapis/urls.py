from django.urls import path
from . import views

urlpatterns=[
    path('',views.askQuestion),
    path('scrape',views.scrape)
]