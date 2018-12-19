from django.shortcuts import render
from api.extract import predict

from django.http import JsonResponse


def index(request):
    return render(request, 'emojify_web/index.html', {'posts': 'vasu'})


def predict_emoji(request):
    text = request.GET.get('emoji_text', None)
    data = predict(text)
    return JsonResponse(data, safe=False)