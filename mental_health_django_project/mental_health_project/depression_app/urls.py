from django.urls import path
from . import views

urlpatterns = [
    path('',              views.dashboard,    name='dashboard'),
    path('analyse/',      views.analyse_view, name='analyse'),
    path('predict/ajax/', views.predict_ajax, name='predict_ajax'),
    path('batch/',        views.batch_upload, name='batch_upload'),
    path('batch/<int:pk>/result/', views.batch_result, name='batch_result'),
    path('history/',      views.history,      name='history'),
    path('train/',        views.train_model,  name='train_model'),
]
