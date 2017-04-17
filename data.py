import os
import random
import re
import time
import math
import sys
import numpy as np

import config

def update_progress(progress, total, cells=30):
    progress += 1
    percent = math.ceil((progress * 100)/total)
    map_ = math.ceil((progress / total) * cells)
    sys.stdout.write('\r[{0}] {1}%'.format('#'*map_, percent))
    sys.stdout.flush()

def get_lines():
    print("Creating line dictionary...")
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.split(b' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == 10:
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line

def get_convos():
    print("Retrieving Conversations...")
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    convos = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.split(b' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(b', '):
                    convo.append(line[1:-1])
                convos.append(convo)
    return convos


def question_answers(id2line, convos):
    print("Gathering question-answer pairs...")

    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers

def prepare_dataset(questions, answers):
    print("Writing q-a pairs to file...")
    make_dir(config.PROCESSED_PATH)

    test_ids = random.sample([i for i in range(len(questions))], config.TESTSET_SIZE)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []

    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename),'wb'))

    for i in range(len(questions)):
        update_progress(i, len(questions))
        if i in test_ids:
            files[2].write(questions[i] + b'\n')
            files[3].write(answers[i] + b'\n')
        else:
            files[0].write(questions[i] + b'\n')
            files[1].write(answers[i] + b'\n')

    for file in files:
        file.close()

    print(' ')

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def prepare_raw_data():
    print("Preparing raw data...")
    id2line = get_lines()
    convos = get_convos()
    questions, answers = question_answers(id2line, convos)
    prepare_dataset(questions, answers)

def process_data():
    print("Preparing data to be model ready...")
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

def basic_tokenizer(line, normalize_digits=True):
    line = re.sub(b'[^\x00-\x7F]+',b'', line)
    line = re.sub(b'<u>', b'', line)
    line = re.sub(b'</u>', b'', line)
    line = re.sub(b'\[', b'', line)
    line = re.sub(b'\]', b'', line)
    
    words = []
    _WORD_SPLIT = re.compile(b"([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(b"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, b'#', token)
            words.append(token)
    return words

def build_vocab(filename):
    print("Building vocab list for {}".format(filename))

    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{0}'.format(filename[-3:]))
    
    vocab = {}
    with open(in_path, 'rb') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    
    with open(out_path, 'wb') as f:
        f.write(b'<PAD>' + b'\n')
        f.write(b'<UNK>' + b'\n')
        f.write(b'<GO>' + b'\n')
        f.write(b'<EOS>' + b'\n') 
        index = 4

        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                break
            f.write(word + b'\n')
            index += 1

def token2id(data, mode):
    print("Tokenizing {0}.{1}...".format(data, mode))
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'rb')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'wb')

    lines = in_file.read().splitlines()
    for i, line in enumerate(lines):
        update_progress(i, len(lines))

        if mode == 'dec':
            ids = [vocab[b'<GO>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == 'dec':
            ids.append(vocab[b'<EOS>'])
        out_file.write(b' '.join(str(id_).encode('utf-8') for id_ in ids) + b'\n')

    print(' ')


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab[b'<UNK>']) for token in basic_tokenizer(line)]

def load_data():
    return process_file('train_ids.enc'),process_file('train_ids.dec'),process_file('test_ids.enc'),process_file('test_ids.enc')

def process_file(file_):
    file_path = os.path.join(config.PROCESSED_PATH, file_)

    processed_file = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            processed_file.append(list(map(int, line.split())))
            
    return processed_file

def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield np.array(pad_sentence_batch(source_batch)), np.array(pad_sentence_batch(target_batch))


def pad_sentence_batch(sentence_batch):
    """
    Pad sentence with <PAD> id
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [0] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


def run():
    if not os.path.isdir("./" + config.PROCESSED_PATH):
        start_time = time.time()
        prepare_raw_data()
        process_data()
        print("Done in {0} seconds.".format(time.time() - start_time))
    else:
        print("Processing already completed!")

if __name__ == '__main__':
    run()