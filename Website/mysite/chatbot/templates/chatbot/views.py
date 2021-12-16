from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from chatbot.utils.seq2seq import Chatbot, initialize, chatOnce

from .models import Input, Response, Log

# Create your views here.
def index(request):
    en, de, se, voc = initialize()
    bot = Chatbot(en, de, se, voc)
    
    latest_conv_list = Log.objects.order_by('-id')[:20]
    #questions=None
    if request.GET.get('context'):
        inp = request.GET.get('context')
        output = chatOnce(bot, inp)
        inpQ = Input(input_text=inp)
        inpQ.save()
        resQ = Response(context=inpQ, response_text=output)
        resQ.save()
        logQ = Log(context=inpQ, response=resQ)
        logQ.save()
        '''search = request.GET.get('search')
        questions = Queries.objects.filter(query__icontains=search)

        name = request.GET.get('name')
        query = Queries.object.create(query=search, user_id=name)
        query.save()
        '''
    if request.POST:
        r = Log.objects.filter()
        r.delete()

    return render(request, 'chatbot/index.html', {
        'latest_conv_list': latest_conv_list
    })