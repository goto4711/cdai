"""llm_tools: small helpers for research with language models.
Cultural Data Analysis, University of Amsterdam. Free to use and change in your own projects.

In Colab:
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/goto4711/cdai/main/llm_tools.py', 'llm_tools.py')
    from llm_tools import *

Keys go into Colab's secrets (the key symbol on the left, switch on "Notebook access"), never into your code:
    GEMINI_API_KEY       free at aistudio.google.com, for model names that start with 'gemini'
    OPENROUTER_API_KEY   openrouter.ai (paid), for names like 'openai/gpt-4.1-mini' or 'anthropic/claude-haiku-4.5'
    ANTHROPIC_API_KEY, OPENAI_API_KEY   if you use these providers directly ('claude-...', 'gpt-...')

Choosing a model
    openrouter_models(search)                        the models on OpenRouter, with their prices (no key needed)
    has_key(model), key_name(model)                  is the key for this model there, and what is it called?
Asking
    ask(prompt, model, temperature)                  one prompt, one answer
    collect(prompts, models, runs, file)             many answers, saved in a CSV file with model, settings and date
Reading the answers
    parse_json(text)                                 the JSON in an answer
    parse_dates(text)                                (year, event) pairs from a "YEAR | EVENT" list
Coding texts with a model, and checking it
    coding_prompt(instructions, categories)          the prompt that code_with_llm() sends
    code_with_llm(texts, instructions, categories)   let a model code (annotate) texts
    validation_sample(df, n)                         a random sample to code by hand
    agreement(a, b)                                  how far two codings agree: percent and Cohen's kappa
Comparing
    shares(df, rows, columns)                        a crosstab as shares of each row
    jaccard(a, b), within_between(sets, labels), jaccard_matrix(sets, labels)
    brier(probability, outcome)                      score probability forecasts (lower is better)
"""
import datetime
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

MODEL = 'gemini-3.5-flash-lite'   # the default: a small Gemini model with a free tier (October 2026)


# ---------------------------------------------------------------------------------------------
# Keys and requests
# ---------------------------------------------------------------------------------------------

def get_key(name):
    """Read a key from Colab's secrets, or else from an environment variable."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def key_name(model=MODEL):
    """The name of the secret that this model needs, e.g. 'OPENROUTER_API_KEY' for 'openai/gpt-4.1-mini'."""
    return _provider(model)[1]


def has_key(model=MODEL):
    """True if the key for this model is there."""
    return bool(get_key(key_name(model)))


def openrouter_models(search=None):
    """The models that OpenRouter offers now, with their prices in dollars per million tokens.
    search: only models whose name contains this text, e.g. 'mistral' or 'gpt'. No key needed.
    Use the 'model' column as the model name in ask() and collect()."""
    r = requests.get('https://openrouter.ai/api/v1/models', timeout=60)
    r.raise_for_status()
    rows = []
    for m in r.json()['data']:
        price = m.get('pricing', {})
        rows.append({'model': m['id'], 'name': m.get('name', ''),
                     'price_in': float(price.get('prompt') or 0) * 1e6, 'price_out': float(price.get('completion') or 0) * 1e6,
                     'context': m.get('context_length'),
                     'added': datetime.date.fromtimestamp(m['created']).isoformat() if m.get('created') else ''})
    df = pd.DataFrame(rows)
    if search:
        df = df[df['model'].str.contains(search, case=False) | df['name'].str.contains(search, case=False)]
    return df.sort_values('model').reset_index(drop=True)


class Stop(Exception):
    """A mistake in the request (a wrong model name, a wrong key ...): trying again won't help."""


def _provider(model):
    """Which provider a model name belongs to, and the name of its key."""
    if '/' in model:                            # 'openai/gpt-4.1-mini': OpenRouter's names
        return 'openrouter', 'OPENROUTER_API_KEY'
    if model.startswith(('gemini', 'gemma')):
        return 'gemini', 'GEMINI_API_KEY'
    if model.startswith('claude'):
        return 'anthropic', 'ANTHROPIC_API_KEY'
    return 'openai', 'OPENAI_API_KEY'


def _post(url, headers, body):
    try:
        r = requests.post(url, headers=headers, json=body, timeout=180)
    except requests.RequestException as e:
        raise RuntimeError(f'no connection ({e.__class__.__name__})')   # worth trying again
    if r.status_code == 429 or r.status_code >= 500:
        raise RuntimeError(f'{r.status_code}: busy')                    # worth trying again
    if r.status_code != 200:
        raise Stop(f'{r.status_code}: {r.text[:300]}')
    return r.json()


def _request(prompt, model, temperature, max_tokens, system, extra, key):
    """One request in the provider's own format. Returns a dictionary with the text and the details."""
    provider, _ = _provider(model)
    if provider == 'gemini':
        config = {'temperature': temperature, **({'maxOutputTokens': max_tokens} if max_tokens else {}), **extra}
        body = {'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': config}
        if system:
            body['systemInstruction'] = {'parts': [{'text': system}]}
        j = _post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                  {'Content-Type': 'application/json', 'x-goog-api-key': key}, body)
        if not j.get('candidates'):                         # e.g. the prompt was blocked
            raise Stop(f"no answer: {j.get('promptFeedback', j)}")
        cand = j['candidates'][0]
        parts = cand.get('content', {}).get('parts', [])
        u = j.get('usageMetadata', {})
        return {'text': ''.join(p.get('text', '') for p in parts if not p.get('thought')),   # leave out the 'thinking'
                'tokens_in': u.get('promptTokenCount'),
                'tokens_out': (u.get('candidatesTokenCount') or 0) + (u.get('thoughtsTokenCount') or 0),
                'finish': cand.get('finishReason'), 'served_by': 'Google', 'raw': j}
    if provider == 'anthropic':
        body = {'model': model, 'max_tokens': max_tokens or 4000, 'temperature': temperature,
                'messages': [{'role': 'user', 'content': prompt}], **extra}
        if system:
            body['system'] = system
        j = _post('https://api.anthropic.com/v1/messages',
                  {'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}, body)
        return {'text': ''.join(b.get('text', '') for b in j['content']),
                'tokens_in': j['usage']['input_tokens'], 'tokens_out': j['usage']['output_tokens'],
                'finish': j.get('stop_reason'), 'served_by': 'Anthropic', 'raw': j}
    # OpenAI, and the APIs that copy its format (OpenRouter and many others)
    url = 'https://openrouter.ai/api/v1' if provider == 'openrouter' else 'https://api.openai.com/v1'
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    body = {'model': model, 'temperature': temperature, 'messages': messages, **extra}
    if max_tokens:
        body['max_tokens'] = max_tokens
    j = _post(url + '/chat/completions', {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, body)
    choice = j['choices'][0]
    return {'text': choice['message'].get('content') or '',
            'tokens_in': j['usage']['prompt_tokens'], 'tokens_out': j['usage']['completion_tokens'],
            'finish': choice.get('finish_reason'), 'served_by': j.get('provider', 'OpenAI'), 'raw': j}


def ask(prompt, model=MODEL, temperature=1.0, max_tokens=None, system=None, details=False, extra=None, tries=4):
    """Send one prompt to a model and return the text of its answer.

    model        'gemini-...' (Google, free key), 'provider/name' (OpenRouter), 'claude-...' or 'gpt-...'
    temperature  randomness: 1.0 is the usual default; lower gives more similar answers (not always identical ones)
    max_tokens   the longest answer you pay for; None = the provider's default. It includes 'thinking' tokens.
    system       an optional system prompt: an instruction that holds for the whole conversation
    details      True: return a dictionary with text, tokens, finish reason, provider and the whole raw answer
    extra        settings added to the request, e.g. {'reasoning': {'effort': 'low'}} on OpenRouter
    tries        how often to try again when the server is busy (waiting 15, 30, 45 ... seconds)
    """
    provider, key_name = _provider(model)
    key = get_key(key_name)
    if not key:
        raise Stop(f"No key: add {key_name} to Colab's secrets (key symbol on the left) and switch on Notebook access.")
    for attempt in range(tries):
        try:
            res = _request(prompt, model, temperature, max_tokens, system, extra or {}, key)
            break
        except Stop:
            raise
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f'  {e}; trying again in {15 * (attempt + 1)} seconds')
            time.sleep(15 * (attempt + 1))
    if str(res['finish']).lower() in ('max_tokens', 'length'):
        print(f'  Warning: the answer of {model} was cut off at max_tokens.')
    if details:
        return {'model': model, 'provider': provider, 'temperature': temperature, **res}
    return res['text']


def collect(prompts, models=MODEL, runs=1, temperature=1.0, file='llm_answers.csv', system=None,
            max_tokens=None, extra=None, keep_raw=False, pause=1):
    """Ask every model every prompt `runs` times and save each answer in a CSV file right away.

    prompts  one prompt, a list of prompts, or a dictionary {name: prompt}, e.g. {'en': '...', 'nl': '...'}
    models   one model name or a list, e.g. ['gemini-3.5-flash-lite', 'openai/gpt-4.1-mini']
    file     the CSV file. Answers already in it (same model, prompt, run and settings) are not asked again: run
             the cell again after an interruption and it goes on where it stopped. New answers are added to the file.
    keep_raw True: also save the whole raw answer of the API (as JSON text) in the column 'raw'

    Every row says when it was asked, which model with which settings, and how many tokens it used:
    report these in your project. Returns the answers to this call (from the file) as a DataFrame.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, dict):
        prompts = {f'p{i}': p for i, p in enumerate(prompts, 1)}
    models = [models] if isinstance(models, str) else list(models)
    def keys(df):   # what makes two requests the same
        return list(zip(df['model'], df['prompt'], df['system'].fillna('').astype(str), df['run'],
                        df['temperature'], df['max_tokens'].fillna(0), df['extra'].fillna('{}')))
    settings = (system or '', temperature, max_tokens or 0, json.dumps(extra or {}))
    wanted = {(m, prompts[pid], settings[0], run, *settings[1:]) for m in models for pid in prompts
              for run in range(1, runs + 1)}
    done = set()
    if os.path.exists(file):
        old = pd.read_csv(file)
        done = set(keys(old)) & wanted
        print(f'{file} already has {len(old)} answers, {len(done)} of them to these requests.')
    todo = [(m, pid, run) for m in models for pid in prompts for run in range(1, runs + 1)
            if (m, prompts[pid], settings[0], run, *settings[1:]) not in done]
    print(f'{len(todo)} requests to send.')
    stopped = set()
    for i, (m, pid, run) in enumerate(todo, 1):
        if m in stopped:
            continue
        try:
            res = ask(prompts[pid], model=m, temperature=temperature, max_tokens=max_tokens, system=system,
                      details=True, extra=extra)
        except Stop as e:
            print(f'{m}: STOPPED - {e}\n  Check the model name and the key; the other models go on.')
            stopped.add(m)
            continue
        except Exception as e:
            print(f'{m}, {pid}, run {run}: no answer ({e}). Run collect() again later to fill the gap.')
            continue
        row = {'date': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               'model': m, 'provider': res['provider'], 'served_by': res['served_by'],
               'prompt_id': pid, 'prompt': prompts[pid], 'system': system or '', 'run': run,
               'temperature': temperature, 'max_tokens': max_tokens, 'extra': json.dumps(extra or {}),
               'text': res['text'], 'tokens_in': res['tokens_in'], 'tokens_out': res['tokens_out'],
               'finish': res['finish'], 'raw': json.dumps(res['raw'], ensure_ascii=False) if keep_raw else ''}
        new = not os.path.exists(file)
        columns = list(row) if new else list(pd.read_csv(file, nrows=0).columns)   # the columns of the file
        pd.DataFrame([row]).reindex(columns=columns).to_csv(file, mode='a', header=new, index=False)
        if i % 10 == 0 or i == len(todo):
            print(f'  {i}/{len(todo)}')
        time.sleep(pause)
    if not os.path.exists(file):
        return pd.DataFrame()
    df = pd.read_csv(file)
    df = df[[k in wanted for k in keys(df)]].reset_index(drop=True)
    print(f'{len(df)} answers to these requests, saved in {file}. In Colab, download the file or save it to '
          'Google Drive: files in the Colab session disappear when it ends.')
    return df


# ---------------------------------------------------------------------------------------------
# Reading the answers
# ---------------------------------------------------------------------------------------------

def parse_json(text):
    """Find the JSON in a model's answer (models often add ```json ... ``` or a sentence around it).
    Returns a list or a dictionary, or None if there is no readable JSON."""
    text = re.sub(r'```(json)?', '', text)
    starts = sorted(i for i in (text.find('['), text.find('{')) if i >= 0)
    for start in starts:
        end = text.rfind(']' if text[start] == '[' else '}')
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            continue
    return None


BCE = r'(BCE|B\.?\s?C\.?|v\.\s?Chr\.?|voor Christus|公元前|紀元前|前(?=\s*\d)|기원전)'


def parse_dates(text):
    """Turn an answer in the format "YEAR | EVENT" (one per line) into a list of (year, event) pairs.
    Also reads "1492 - event", "1492: event" and markdown tables.
    Years before the common era become negative numbers, e.g. 221 BCE -> -221. Centuries are left out."""
    dates = []
    for line in text.splitlines():
        line = line.replace('*', '').replace('｜', '|').replace('：', ':').strip()
        line = line.strip('|').strip()                       # rows of a markdown table
        if not re.search(r'\d', line) or re.fullmatch(r'[|:\-\s]+', line):
            continue
        if '|' in line:
            cells = [c.strip() for c in line.split('|')]
            # the year is in the first cell with a number (skip a column that only numbers the rows)
            i = next((k for k, c in enumerate(cells) if re.search(r'\d{3}|\d\s*(BCE|BC|v\.\s?Chr|年)', c)),
                     next((k for k, c in enumerate(cells) if re.search(r'\d', c)), 0))
            left, event = cells[i], ' | '.join(c for c in cells[i + 1:] if c)
        else:                                                # e.g. "1492 - Columbus" or "1492: Columbus"
            m = re.match(r'^(.{0,30}?\d+[^\s:–—-]*)\s*(?::\s*|[–—-]\s+)(.+)$', line)
            if not m:
                continue
            left, event = m.group(1), m.group(2)
        if re.search(r'century|eeuw|世纪|世紀', left, re.I):  # "3rd century BCE": no single year
            continue
        left = re.sub(r'^\d{1,2}[.)]\s+(?=.*\d)', '', left.strip())   # a list number like "3. 1492"
        left = left.replace(',', '')
        nums = re.findall(r'-?\d{1,5}', left)
        if not nums or not event:
            continue
        year = int(next((n for n in nums if len(n.lstrip('-')) >= 3), nums[0]))   # "4 July 1776" -> 1776
        if year > 0 and re.search(BCE, left, re.I):
            year = -year
        dates.append((year, event.strip()))
    return dates


# ---------------------------------------------------------------------------------------------
# Coding texts with a model, and checking it
# ---------------------------------------------------------------------------------------------

def coding_prompt(instructions, categories):
    """The prompt that code_with_llm() sends, without the texts. Print it, and report it in your project.

    instructions  what the texts are and how to code them, in your own words
    categories    {'column name': [allowed values]}, e.g. {'region': ['Europe', 'Africa', ...]}.
                  Use None instead of a list for a free answer, e.g. {'place': None}.
    """
    lines = [instructions.strip(), '', 'For each item, give:']
    for name, values in categories.items():
        lines.append(f'- "{name}": ' + ('one of: ' + ', '.join(values) + '.' if values else 'your answer, in a few words.'))
    keys = ', '.join(f'"{k}"' for k in ['id', *categories])
    lines += [f'Answer with JSON only: a list of objects with the keys {keys}. No other text.', '', 'Items:']
    return '\n'.join(lines)


def code_with_llm(texts, instructions, categories, model=MODEL, batch_size=40, temperature=0, file=None, **ask_kw):
    """Let a model code texts: it puts each text into one of your categories.

    texts       a list or a column of texts. Each distinct text is coded once.
    instructions, categories   see coding_prompt()
    batch_size  texts per request: fewer requests, but long batches are coded less carefully
    temperature 0 by default: as similar as possible each time (still not guaranteed identical)
    file        optional CSV file to save the codes in
    Returns a DataFrame with the column 'text', one column per category, and 'model'.
    A value that is missing or not in your list becomes 'unclear': look at these.
    """
    prompt = coding_prompt(instructions, categories)
    distinct = list(dict.fromkeys(str(t) for t in texts))
    codes = {}
    for start in range(0, len(distinct), batch_size):
        batch = distinct[start:start + batch_size]
        listing = '\n'.join(f'{i}: {t}' for i, t in enumerate(batch))
        items = None
        for attempt in range(3):
            items = parse_json(ask(prompt + '\n' + listing, model=model, temperature=temperature, **ask_kw))
            if isinstance(items, list):
                break
            print(f'  batch {start}: no readable JSON, trying again')
        for it in items if isinstance(items, list) else []:
            try:
                text = batch[int(it['id'])]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            codes[text] = {name: (it.get(name) if (values is None and it.get(name)) or (values and it.get(name) in values)
                                  else 'unclear') for name, values in categories.items()}
        print(f'{min(start + batch_size, len(distinct))}/{len(distinct)} coded')
    rows = [{'text': t, **codes.get(t, {name: 'unclear' for name in categories}), 'model': model} for t in distinct]
    df = pd.DataFrame(rows)
    df.attrs['prompt'] = prompt
    missing = (df[list(categories)] == 'unclear').any(axis=1).sum()
    if missing:
        print(f'{missing} texts with "unclear" codes: look at them, e.g. df[df["{next(iter(categories))}"] == "unclear"]')
    if file:
        df.to_csv(file, index=False)
        print('saved', file)
    return df


def validation_sample(df, n=20, text='text', seed=4711):
    """Draw a random sample to code by hand, and print its texts with numbers.
    Code them before you look at the model's codes. The same seed gives the same sample again."""
    sample = df.drop_duplicates(text).sample(min(n, df[text].nunique()), random_state=seed).reset_index(drop=True)
    for i, t in enumerate(sample[text]):
        print(f'{i}: {t}')
    return sample


def agreement(a, b):
    """How far two codings of the same items agree, e.g. yours and the model's.

    percent  the share of items with the same code
    kappa    Cohen's kappa: the agreement beyond what chance would give. 1 = perfect, 0 = no better than chance.
             A common rule of thumb: above 0.8 is very good, 0.6-0.8 good, below 0.4 weak.
    """
    a, b = [str(x) for x in a], [str(x) for x in b]
    if len(a) != len(b):
        raise ValueError(f'The two codings have different lengths: {len(a)} and {len(b)}.')
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)   # agreement by chance
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return pd.Series({'items': n, 'percent': round(observed, 3), 'kappa': round(kappa, 3)}, dtype=object)


# ---------------------------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------------------------

def shares(df, rows, columns):
    """A crosstab of two columns, as the share of each row (each row adds up to 1)."""
    return pd.crosstab(df[rows], df[columns], normalize='index')


def jaccard(a, b):
    """The overlap of two sets: the number in both, divided by the number in either. 1 = the same, 0 = nothing shared."""
    a, b = set(a), set(b)
    if not a | b:
        return 0.0
    return len(a & b) / len(a | b)


def within_between(sets, labels):
    """The average Jaccard index of all pairs with the same label (e.g. the same model)
    and of all pairs with different labels. Returns (within, between)."""
    sets, labels = list(sets), list(labels)
    within, between = [], []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            (within if labels[i] == labels[j] else between).append(jaccard(sets[i], sets[j]))
    return float(np.mean(within)) if within else np.nan, float(np.mean(between)) if between else np.nan


def jaccard_matrix(sets, labels):
    """The average Jaccard index for every pair of labels, e.g. models. The diagonal compares
    different answers with the same label. Plot it with seaborn's heatmap()."""
    sets, labels = list(sets), list(labels)
    names = sorted(set(labels))
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for x in names:
        for y in names:
            scores = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(len(sets))
                      if i != j and labels[i] == x and labels[j] == y]
            m.loc[x, y] = np.mean(scores) if scores else np.nan
    return m


def brier(probability, outcome):
    """The Brier score of probability forecasts: (probability - outcome) squared, for each forecast.
    outcome is 1 if it happened and 0 if not. Take the mean for an overall score.
    0 = perfect; always saying 50% gives 0.25; lower is better."""
    p, o = np.asarray(probability, dtype=float), np.asarray(outcome, dtype=float)
    score = (p - o) ** 2
    return pd.Series(score, index=probability.index) if isinstance(probability, pd.Series) else score
