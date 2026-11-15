"""TypeSafe System One advisory comparison, never an execution authority."""
import json
import math
import os
import urllib.request

QUESTIONS = {
    'reminder_needed': {'type': 'noul', 'instructions':
        'Does the supplied evidence justify a stand-steady reminder now? Normal rising alone '
        'does not establish danger. Consider observation quality. Missing data is unknown.'},
    'fall_review_needed': {'type': 'noul', 'instructions':
        'Does the supplied observed evidence warrant human review of a possible fall? '
        'A model score alone is not confirmation. This advice cannot change alarms.'},
}


def assess(request, *, enabled=False, timeout=5, transport=None):
    if not enabled:
        return dict(status='NOT_RUN', reason='comparison disabled', changes_alarm=False)
    token = os.environ.get('TYPESAFE_API_KEY')
    if not token and transport is None:
        return dict(status='UNAVAILABLE', reason='TYPESAFE_API_KEY missing', changes_alarm=False)
    question = 'reminder_needed' if request['state']['branch'] == 'reminder' else 'fall_review_needed'
    body = dict(model=os.environ.get('JEV_MODEL', 'jev-latest'), state=request['state'],
                questions={question: QUESTIONS[question]})
    try:
        if transport is None:
            req = urllib.request.Request('https://api.typesafe.ai/v1/systemone',
                data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {token}',
                                                       'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = json.load(response)
        else:
            raw = transport(body)
        answer = raw['answers'][question]
        score = answer['noul']
        if (answer['type'] != 'noul' or type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Invalid answer')
        return dict(status='OK', question=question, score=score, model=raw['model'],
                    changes_alarm=False, score_semantics='model judgment, not calibrated fall risk')
    except Exception as error:
        return dict(status='UNAVAILABLE', reason=type(error).__name__, changes_alarm=False)
