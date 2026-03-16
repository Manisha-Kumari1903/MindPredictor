from django.urls import path
from . import views

urlpatterns = [
    path('',                views.login_view,      name='login'),
    path('login/',          views.login_view,       name='login'),
    path('register/',       views.register_view,    name='register'),
    path('logout/',         views.logout_view,      name='logout'),
    path('dashboard/',      views.dashboard,        name='dashboard'),
    path('analyse/',        views.analyse_view,     name='analyse'),
    path('depressive-rate/', views.depressive_rate, name='depressive_rate'),
]
