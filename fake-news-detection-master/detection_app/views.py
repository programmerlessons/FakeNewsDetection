from django.http import HttpResponse
from .mlmodel import fake as f
from django.http import JsonResponse
import json
from django.shortcuts import render
def index(request):
    if request.method == 'POST':
        data=request.POST.get("news")
        print(data)
        result = int(f.find(data))
        print(result)
        return JsonResponse({'result':result})
    return render(request,'detection_app/index.html')



    