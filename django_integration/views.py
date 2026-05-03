import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services import translator

def index(request):
    """Renders the main translation UI"""
    context = {
        'bleu_score': 38.36,
        'meteor_score': 0.6043,
        'bert_score': 0.9123,
        'comet_score': 0.8848
    }
    return render(request, 'mt_app/index.html', context)

@csrf_exempt
def translate_api(request):
    """API endpoint to handle AJAX translation requests"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            source_text = data.get('text', '')
            model_choice = data.get('model', 'nllb')
            
            if not source_text:
                return JsonResponse({'error': 'No text provided'}, status=400)
                
            translation = translator.translate(source_text, model_name=model_choice)
            
            return JsonResponse({
                'translation': translation,
                'model_used': model_choice
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)
