"""
Personal data for the CDA Individual Miniproject (2026/27) and its resit.

Every student gets their own version of the exercise data, made from their student number.
The tasks are the same for everyone; only the numbers (and so the answers) differ.

In the notebook (one given cell at the top):
    from cda_personal import make_my_data
    make_my_data(STUDENT_NUMBER)                         # individual miniproject
    make_my_data(STUDENT_NUMBER, assignment='resit')     # resit

Week 1 uses a plain Python list instead of a file:
    from cda_personal import my_playlist      (resit: my_booklist)
    playlist = my_playlist(STUDENT_NUMBER)

The expected answers for grading are in a separate file that is not given to students
(cda_answers.py, in the grading folder).

The data only depend on random.random(), whose sequence Python guarantees not to change
between versions, so students, Colab and the grading script always see exactly the same data.
"""
import csv
import math
import os
import random

COURSE = 'cda-2026'   # change every year: last year's data and answers then no longer fit

FILES = {
    'individual': ['exhibition_posts.csv', 'library_members.csv'],
    'resit': ['festival_posts.csv', 'concert_audience.csv'],
}


# ---------------------------------------------------------------- helpers (random.random only)
def _check(number):
    n = str(number).strip()
    if not n.isdigit() or not 6 <= len(n) <= 9:
        raise ValueError('Please type your student number between the quotes, e.g. STUDENT_NUMBER = "12345678"')
    return n


def _rng(number, what, attempt=0):
    return random.Random('%s|%s|%s|%d' % (COURSE, what, number, attempt))


def _u(r, a, b):
    return a + (b - a) * r.random()


def _i(r, a, b):
    """Whole number from a to b (both included)."""
    return a + int(r.random() * (b - a + 1))


def _normal(r, mean, sd):
    u1 = max(r.random(), 1e-12)
    return mean + sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * r.random())


def _shuffle(r, items):
    items = list(items)
    for k in range(len(items) - 1, 0, -1):
        j = int(r.random() * (k + 1))
        items[k], items[j] = items[j], items[k]
    return items


def _choice(r, items):
    return items[int(r.random() * len(items))]


def _write(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for x in rows:
            w.writerow(['' if x.get(c) is None else x[c] for c in columns])


def _top_two(counts):
    s = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return s[0], (s[1] if len(s) > 1 else (None, 0))


# ================================================================ WEEK 1: plain Python lists
SONG_WORDS = ['Blue', 'Midnight', 'River', 'Golden', 'Echo', 'Paper', 'Silver', 'Summer', 'City', 'Wild',
              'Neon', 'Quiet', 'Velvet', 'Northern', 'Electric', 'Lonely', 'Crystal', 'Broken', 'Endless', 'Hidden']
SONG_NOUNS = ['Heart', 'Lights', 'Road', 'Dreams', 'Rain', 'Garden', 'Sky', 'Waves', 'Fire', 'Station',
              'Letters', 'Morning', 'Harbour', 'Shadows', 'Tide', 'Avenue', 'Mirror', 'Song', 'Window', 'Stars']
ARTISTS = ['The Marigolds', 'Nova Lane', 'Kees & the Keys', 'Laila Ortega', 'Fjord', 'Amira Haddad', 'The Night Owls',
           'Jun Park', 'Delta Bloom', 'Otis Grey', 'Sofia Brandt', 'Mango Season', 'Ruben Kort', 'Echo Valley']


def my_playlist(number):
    """Week 1 (individual): the student's playlist as a list of dictionaries."""
    number = _check(number)
    for attempt in range(200):
        r = _rng(number, 'playlist', attempt)
        n = _i(r, 18, 24)
        fav = _choice(r, [1960, 1970, 1980, 1990, 2000, 2010, 2020])
        songs, titles = [], set()
        for k in range(n):
            decade = fav if k < 6 else _choice(r, [1960, 1970, 1980, 1990, 2000, 2010, 2020])
            year = min(2025, decade + _i(r, 0, 9))
            title = '%s %s' % (_choice(r, SONG_WORDS), _choice(r, SONG_NOUNS))
            if title in titles:
                continue
            titles.add(title)
            songs.append({'title': title, 'artist': _choice(r, ARTISTS), 'year': year, 'seconds': _i(r, 140, 330)})
        songs = _shuffle(r, songs)
        counts = {}
        for s in songs:
            counts[s['year'] // 10 * 10] = counts.get(s['year'] // 10 * 10, 0) + 1
        (d1, c1), (d2, c2) = _top_two(counts)
        if c1 >= c2 + 2:
            return songs
    raise RuntimeError('could not make a playlist')


BOOK_WORDS = ['The Silent', 'A Winter', 'The Last', 'Under the', 'The Glass', 'Letters from the', 'The Long',
              'A House of', 'Beyond the', 'The Paper', 'Night in the', 'The Salt']
BOOK_NOUNS = ['Garden', 'Archive', 'Harbour', 'Orchard', 'Lighthouse', 'Map', 'River', 'Library', 'Island',
              'Museum', 'Forest', 'Atlas', 'Bridge', 'Theatre']
AUTHORS = ['M. de Wit', 'A. Nakamura', 'E. Okafor', 'L. Moreau', 'S. Lindqvist', 'R. Haddad', 'J. Kowalski',
           'T. Brennan', 'F. Rossi', 'H. Bakker', 'C. Mendes', 'Y. Chen']


def my_booklist(number):
    """Week 1 (resit): the student's reading list as a list of dictionaries."""
    number = _check(number)
    for attempt in range(200):
        r = _rng(number, 'books', attempt)
        n = _i(r, 15, 20)
        fav = _choice(r, ['short', 'medium', 'long'])
        span = {'short': (90, 199), 'medium': (200, 400), 'long': (401, 780)}
        books, titles = [], set()
        for k in range(n):
            cat = fav if k < 6 else _choice(r, ['short', 'medium', 'long'])
            title = '%s %s' % (_choice(r, BOOK_WORDS), _choice(r, BOOK_NOUNS))
            if title in titles:
                continue
            titles.add(title)
            books.append({'title': title, 'author': _choice(r, AUTHORS), 'year': _i(r, 1950, 2025),
                          'pages': _i(r, *span[cat])})
        books = _shuffle(r, books)
        counts = {}
        for b in books:
            counts[_length_class(b['pages'])] = counts.get(_length_class(b['pages']), 0) + 1
        (c1n, c1), (c2n, c2) = _top_two(counts)
        longest = sorted(books, key=lambda b: -b['pages'])
        if c1 >= c2 + 2 and longest[0]['pages'] > longest[1]['pages']:
            return books
    raise RuntimeError('could not make a book list')


def _length_class(pages):
    return 'short' if pages < 200 else ('medium' if pages <= 400 else 'long')


# ================================================================ WEEK 2: social media posts
EXHIBITION = {
    'campaign': '#HiddenHarbours',
    'accounts': ['harbour_mia', 'museumgoer_tom', 'ams_history', 'lena_sketches', 'docklands_dad', 'photo_ruben',
                 'curious_kai', 'maritime_nerd', 'noor_visits', 'artbus_ams', 'oldmaps_jan', 'weekend_eva'],
    'tags': ['#MuseumNight', '#Maritime', '#OpenDepot', '#FamilyDay', '#VintageMaps', '#Photography',
             '#Archives', '#CityHistory'],
    'texts': ['Just walked through {x} at the city museum - stunning ship models!',
              'The old harbour maps in {x} are incredible.',
              'Took the kids to {x} today, they loved the diving suits.',
              'Not sure about the lighting in {x}, but the objects are great.',
              'Third visit to {x}. Still finding new details.',
              'The audio guide for {x} is worth it.',
              'Queue was long, but {x} made up for it.',
              'Loved the photographs of dock workers in {x}.'],
    'name': 'Hidden Harbours',
    'start': (2026, 3, 2),
}
FESTIVAL = {
    'campaign': '#NorthernLightsFF',
    'accounts': ['cinephile_sara', 'filmclub_utrecht', 'reelreviews', 'popcorn_pieter', 'docu_fan', 'arthouse_amira',
                 'shortfilm_jonas', 'festival_diary', 'yusuf_watches', 'lens_and_light', 'critic_kim', 'nightscreen'],
    'films': ['Salt and Silence', 'The Last Ferry', 'Paper Moons', 'Copper Sky', 'Little Thunder'],
    'tags': ['#Cinema', '#WorldPremiere', '#Documentary', '#ShortFilm', '#QandA', '#Arthouse', '#Popcorn', '#Debut'],
    'texts': ['Just saw {f} - what a debut!', '{f} left the whole room silent.', 'Mixed feelings about {f}.',
              'The Q&A after {f} was the best part.', '{f} is my festival favourite so far.',
              'Cinematography in {f} is unreal.', 'Not for everyone, but {f} stayed with me.',
              'Queued an hour for {f}. Worth it.'],
    'start': (2026, 2, 9),
}


def _date(start, day):
    y, m, d = start
    return '%04d-%02d-%02d' % (y, m, d + day)


def exhibition_posts(number):
    """Week 2 Exercise 3 (individual): posts about the museum exhibition 'Hidden Harbours'."""
    number = _check(number)
    S = EXHIBITION
    for attempt in range(200):
        r = _rng(number, 'exhibition', attempt)
        accounts = _shuffle(r, S['accounts'])[:8]
        n = _i(r, 45, 60)
        peak_day = _i(r, 0, 6)
        fav_tag = _choice(r, S['tags'])
        rows = []
        for k in range(n):
            day = peak_day if r.random() < 0.22 else _i(r, 0, 6)
            views = _i(r, 300, 5000)
            rate = _u(r, 0.02, 0.09)
            inter = max(3, int(views * rate))
            likes = int(inter * _u(r, 0.6, 0.8))
            shares = int(inter * _u(r, 0.05, 0.2))
            comments = max(0, inter - likes - shares)
            if day == peak_day:
                likes = int(likes * 1.6)
            tags = [S['campaign']] + ([fav_tag] if r.random() < 0.3 else []) + \
                   ([_choice(r, S['tags'])] if r.random() < 0.5 else [])
            tags = list(dict.fromkeys(tags))
            text = _choice(r, S['texts']).format(x=S['name']) + ' ' + ' '.join(tags)
            rows.append({'account': _choice(r, accounts), 'day': day,
                         'time': (_i(r, 8, 22), _choice(r, [0, 5, 10, 20, 30, 40, 45, 50])),
                         'text': text, 'likes': likes, 'shares': shares, 'comments': comments, 'views': views})
        # one post that stands out by its engagement rate (a small account with a viral post)
        star = rows[_i(r, 0, n - 1)]
        star['views'] = _i(r, 250, 600)
        star['likes'] = int(star['views'] * _u(r, 0.22, 0.3))
        star['shares'] = int(star['views'] * _u(r, 0.04, 0.07))
        star['comments'] = int(star['views'] * _u(r, 0.02, 0.04))
        rows.sort(key=lambda x: (x['day'], x['time']))
        # cleaning traps: 2 posts without text, 3 missing shares (not on the standout post)
        others = [k for k, x in enumerate(rows) if x is not star]
        picks = _shuffle(r, others)
        for k in picks[:2]:
            rows[k]['text'] = None
        for k in picks[2:5]:
            rows[k]['shares'] = None
        out = []
        for k, x in enumerate(rows):
            out.append({'post_id': 1001 + k, 'account': x['account'],
                        'timestamp': '%s %02d:%02d:00' % (_date(S['start'], x['day']), x['time'][0], x['time'][1]),
                        'text': x['text'], 'likes': x['likes'], 'shares': x['shares'], 'comments': x['comments'],
                        'views': x['views']})
        if _ok_exhibition(out):
            return out
    raise RuntimeError('could not make exhibition posts')


def _clean_posts(rows):
    """What the exercise asks: drop posts without text, missing shares count as 0."""
    out = []
    for x in rows:
        if x['text'] is None:
            continue
        y = dict(x)
        y['shares'] = 0 if y['shares'] is None else y['shares']
        out.append(y)
    return out


def _hashtags(rows, campaign):
    counts = {}
    for x in rows:
        for w in x['text'].split():
            if w.startswith('#') and w != campaign:
                counts[w] = counts.get(w, 0) + 1
    return counts


def festival_posts(number):
    """Week 2 Exercise 3 (resit): posts about the Northern Lights Film Festival."""
    number = _check(number)
    S = FESTIVAL
    for attempt in range(200):
        r = _rng(number, 'festival', attempt)
        accounts = _shuffle(r, S['accounts'])[:8]
        star_acc = accounts[0]
        n = _i(r, 45, 60)
        busy_day = _i(r, 0, 6)
        films = _shuffle(r, S['films'])
        hot, loved = films[0], films[1]          # most mentioned / best liked
        rows = []
        for k in range(n):
            day = busy_day if r.random() < 0.25 else _i(r, 0, 6)
            film = hot if r.random() < 0.35 else _choice(r, films)
            views = _i(r, 300, 4000)
            inter = max(3, int(views * _u(r, 0.02, 0.08)))
            likes = int(inter * _u(r, 0.6, 0.8))
            if film == loved:
                likes = int(likes * 2.2) + 40
            shares = int(inter * _u(r, 0.05, 0.2))
            comments = max(0, inter - likes - shares) if film != loved else int(inter * 0.15)
            acc = star_acc if r.random() < 0.2 else _choice(r, accounts[1:])
            if acc == star_acc:
                likes, shares, comments = likes + _i(r, 60, 120), shares + _i(r, 10, 25), comments + _i(r, 10, 25)
            tags = [S['campaign']] + ([_choice(r, S['tags'])] if r.random() < 0.6 else [])
            text = _choice(r, S['texts']).format(f=film) + ' ' + ' '.join(tags)
            rows.append({'account': acc, 'day': day, 'time': (_i(r, 10, 23), _choice(r, [0, 10, 15, 30, 40, 45])),
                         'text': text, 'likes': likes, 'shares': shares, 'comments': comments, 'views': views})
        rows.sort(key=lambda x: (x['day'], x['time']))
        picks = _shuffle(r, [k for k, x in enumerate(rows) if x['account'] != star_acc])
        for k in picks[:2]:
            rows[k]['text'] = None
        for k in picks[2:5]:
            rows[k]['shares'] = None
        out = []
        for k, x in enumerate(rows):
            out.append({'post_id': 2001 + k, 'account': x['account'],
                        'timestamp': '%s %02d:%02d:00' % (_date(S['start'], x['day']), x['time'][0], x['time'][1]),
                        'text': x['text'], 'likes': x['likes'], 'shares': x['shares'], 'comments': x['comments'],
                        'views': x['views']})
        if _ok_festival(out):
            return out
    raise RuntimeError('could not make festival posts')



def _ok_exhibition(rows):
    """Is every question about these posts clearly answerable (no near ties)? Yes or no only."""
    c = _clean_posts(rows)
    rates = sorted((x['likes'] + x['shares'] + x['comments']) / x['views'] for x in c)
    per_day = {}
    for x in c:
        per_day[x['timestamp'][:10]] = per_day.get(x['timestamp'][:10], 0) + x['likes']
    days = sorted(per_day.values())
    tags = sorted(_hashtags(c, EXHIBITION['campaign']).values())
    return rates[-1] > 1.4 * rates[-2] and days[-1] > 1.1 * days[-2] and tags[-1] >= tags[-2] + 2


def _ok_festival(rows):
    """Is every question about these posts clearly answerable (no near ties)? Yes or no only."""
    c = _clean_posts(rows)
    tot, per_day, mentions, likes = {}, {}, {}, {}
    for x in c:
        tot[x['account']] = tot.get(x['account'], 0) + x['likes'] + x['shares'] + x['comments']
        per_day[x['timestamp'][:10]] = per_day.get(x['timestamp'][:10], 0) + 1
        for f in FESTIVAL['films']:
            if f.lower() in x['text'].lower():
                mentions[f] = mentions.get(f, 0) + 1
                likes.setdefault(f, []).append(x['likes'])
    if len(mentions) < len(FESTIVAL['films']) or min(mentions.values()) < 2:
        return False
    t, d, m = sorted(tot.values()), sorted(per_day.values()), sorted(mentions.values())
    a = sorted(sum(v) / len(v) for v in likes.values())
    return t[-1] > 1.15 * t[-2] and d[-1] >= d[-2] + 2 and m[-1] >= m[-2] + 2 and a[-1] > 1.25 * a[-2]


# ================================================================ WEEK 3: clustering
LIBRARY = {   # mean ranges: age, fiction loans, non-fiction loans per year; sd
    'Occasional readers': {'Age': (19, 24, 3), 'Fiction_Loans': (4, 8, 2.5), 'Nonfiction_Loans': (3, 7, 2.5)},
    'Non-fiction learners': {'Age': (33, 39, 4), 'Fiction_Loans': (6, 10, 3), 'Nonfiction_Loans': (36, 46, 4.5)},
    'Fiction fans': {'Age': (46, 54, 5), 'Fiction_Loans': (44, 54, 4.5), 'Nonfiction_Loans': (6, 11, 3)},
    'Heavy all-round readers': {'Age': (64, 72, 5), 'Fiction_Loans': (36, 44, 4.5), 'Nonfiction_Loans': (34, 42, 4.5)},
}
CONCERTS = {   # mean ranges: age, euros per year on classical and on pop concerts; sd
    'Classical regulars': {'Age': (60, 70, 6), 'Spend_Classical': (380, 460, 45), 'Spend_Pop': (20, 60, 20)},
    'Pop fans': {'Age': (21, 28, 3), 'Spend_Classical': (10, 40, 15), 'Spend_Pop': (330, 420, 45)},
    'Omnivores': {'Age': (38, 47, 5), 'Spend_Classical': (180, 240, 35), 'Spend_Pop': (170, 230, 35)},
}


def _clusters(number, what, spec, n_range):
    r = _rng(number, what)
    n = _i(r, *n_range)
    shares = [_u(r, 0.85, 1.15) for _ in spec]
    sizes = [int(n * s / sum(shares)) for s in shares]
    sizes[-1] = n - sum(sizes[:-1])
    rows = []
    for (name, p), size in zip(spec.items(), sizes):
        means = {k: _u(r, v[0], v[1]) for k, v in p.items()}
        for _ in range(size):
            row = {'_group': name}
            for k, v in p.items():
                val = _normal(r, means[k], v[2])
                row[k] = int(round(min(90, max(16, val)))) if k == 'Age' else int(round(max(0, val)))
            rows.append(row)
    return _shuffle(r, rows)


def library_members(number):
    """Week 3 (individual): library members with their age and loans per year."""
    return _clusters(_check(number), 'library', LIBRARY, (360, 400))


def concert_audience(number):
    """Week 3 (resit): concert-goers with their age and yearly spending on concerts."""
    return _clusters(_check(number), 'concerts', CONCERTS, (280, 320))


# ================================================================ for the notebook
def make_my_data(number, folder='.', assignment='individual'):
    """Write the student's own data files for the miniproject (or the resit)."""
    number = _check(number)        # stops with a clear message if no valid number was typed
    os.makedirs(folder, exist_ok=True)
    posts_cols = ['post_id', 'account', 'timestamp', 'text', 'likes', 'shares', 'comments', 'views']
    if assignment == 'resit':
        _write(os.path.join(folder, 'festival_posts.csv'), festival_posts(number), posts_cols)
        _write(os.path.join(folder, 'concert_audience.csv'), concert_audience(number),
               ['Age', 'Spend_Classical', 'Spend_Pop'])
    else:
        _write(os.path.join(folder, 'exhibition_posts.csv'), exhibition_posts(number), posts_cols)
        _write(os.path.join(folder, 'library_members.csv'), library_members(number),
               ['Age', 'Fiction_Loans', 'Nonfiction_Loans'])
    print('✅ Your personal data for student number %s are ready: %s' % (number, ', '.join(FILES[assignment])))


if __name__ == '__main__':
    import sys
    for num in sys.argv[1:] or ['12345678']:
        make_my_data(num, 'example_' + num)
        make_my_data(num, 'example_' + num, 'resit')
