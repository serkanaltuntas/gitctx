"""Native chat API thinking Gemma 4 E4B full-source candidates, never automatic labels."""
from datetime import datetime, timezone
import json
import math

from gitctx.grounded_targets import CONTEXTS, SYSTEM as BASE_SYSTEM
from gitctx.reference_review import request_json, sha
from gitctx.teacher_response import decode_response

VERSION = 'gemma4-e4b-chat-target-candidate-v1'
SYSTEM = BASE_SYSTEM.replace(
    'Return JSON fields type, scope, subject. Use an empty scope for changes across ',
    'Return exactly one plain Conventional Commit header, without JSON or markdown fences. '
    'Omit the scope for changes across '
)


def render(record):
    payload = {'repository': record['source_repo_url'], 'paths': record['changed_paths'],
               'complete_diff': record['diff']}
    # All native markers begin with '<'; JSON escapes round-trip without adding roles.
    content = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    return ('<bos><|turn>system\n<|think|>\n' + SYSTEM + '<turn|>\n<|turn>user\n' + content
            + '<turn|>\n<|turn>model\n')


def generate(record, *, tokenizer, student_tokenizer, model, model_digest,
             model_license, seed=71, temperature=0):
    if record.get('evaluation_only') or record.get('data_split') != 'DEV':
        raise ValueError('only real DEV records can produce training candidates')
    if model_license != 'Apache-2.0':
        raise ValueError('explicit reviewed Apache-2.0 teacher required')
    if not math.isfinite(temperature) or not 0 <= temperature <= 1:
        raise ValueError('invalid temperature')
    prompt = render(record)
    expected = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    context = next((n for n in CONTEXTS if expected + 2064 <= n), None)
    if context is None:
        raise ValueError('complete source exceeds teacher context')
    base = 'http://127.0.0.1:11434/api/'
    tags = request_json(base + 'tags')['models']
    if not any(m['name'] == model and m['digest'] == model_digest for m in tags):
        raise ValueError('teacher digest changed')
    info = request_json(base + 'show', {'model': model})['model_info']
    architecture = info['general.architecture']
    native_context = info.get(architecture + '.context_length')
    if architecture != 'gemma4' or type(native_context) is not int or native_context < context:
        raise ValueError('native Gemma 4 context capacity is unverified or insufficient')
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': prompt.split('<|turn>user\n', 1)[1].split('<turn|>', 1)[0]}]
    chat_response = request_json(base + 'chat', {
        'model': model, 'messages': messages, 'stream': False, 'think': True,
        'keep_alive': '10m',
        'options': {'num_ctx': context, 'num_predict': 2048, 'temperature': temperature,
                    'seed': seed, 'num_thread': 4, 'stop': ['<turn|>', '<eos>']}})
    # Preserve the exact chat response and extract only the API's final content.
    # A thought channel is neither stripped from text nor converted into a label.
    response = {**chat_response, 'response': chat_response['message']['content']}
    fields = target = None
    errors = []
    try:
        fields, target = decode_response(response['response'], 'text')
        if len(student_tokenizer.encode(target)) + 1 > 256:
            errors.append('student answer overflow')
    except (ValueError, TypeError, KeyError):
        errors.append('invalid candidate syntax')
    if response.get('done_reason') != 'stop':
        errors.append('generation did not stop normally')
    if abs(expected - response.get('prompt_eval_count', -10000)) > 2:
        errors.append('prompt token mismatch')
    return {'record_id': record['id'], 'source_diff_sha256': sha(record['diff']),
            'prompt_sha256': sha(prompt), 'prompt_version': VERSION, 'model': model,
            'model_digest': model_digest, 'model_license': model_license,
            'model_architecture': architecture, 'model_context_tokens': native_context,
            'source_format': 'unified', 'output_format': 'text', 'thinking_enabled': True,
            'temperature': temperature, 'seed': seed, 'context_tokens': context,
            'prompt_tokens_expected': expected, 'response': response, 'chat_response': chat_response, 'fields': fields,
            'target': target, 'target_sha256': sha(target) if target else None,
            'target_origin': 'licensed_open_teacher', 'original_reference_in_prompt': False,
            'validation_errors': errors, 'training_approved': False,
            'independent_human_review': False, 'timestamp': datetime.now(timezone.utc).isoformat()}
