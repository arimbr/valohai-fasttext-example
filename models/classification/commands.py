import re
import json
import string
import os
import multiprocessing
import random

import click
import pandas as pd
import fasttext

from utils import get_input_path, get_output_path


FEATURE_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
RANDOM_SEED = 42

train_parameters = {
    'lr': 0.1,
    'dim': 100,
    'ws': 5,
    'epoch': 5,
    'minCount': 1,
    'minCountLabel': 0,
    'minn': 0,
    'maxn': 0,
    'neg': 5,
    'wordNgrams': 1,
    'bucket': 2000000,
    'thread': multiprocessing.cpu_count() - 1,
    'lrUpdateRate': 100,
    't': 1e-4,
    'label': '__label__',
    'verbose': 2,
    'pretrainedVectors': '',
    'seed': 0,
}

def get_model_parameters(model):
    args_getter = model.f.getArgs()

    parameters = {}
    for param in train_parameters:
        attr = getattr(args_getter, param)
        if param == 'loss':
            attr = attr.name
        parameters[param] = attr

    return parameters


def split_text(text):
    text, label = text.split('__label__')
    return text.strip(), label.strip()


def process_string(s):
    # Transform multiple spaces and \n to a single space
    s = re.sub(r'\s{1,}', ' ', s)
    # Remove punctuation
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    s = s.translate(remove_punct_map)
    # Transform to lowercase
    s = s.lower()
    return s


@click.group()
def classification():
    pass


@classification.command()
@click.option('--input_dir', default='input_dir')
@click.option('--output_file', default='output_file')
def collect_bbc_data(input_dir, output_file):

    def rows_generator():
        for root, _, files in os.walk(input_dir):
            category = root.split('/')[-1]
            for fname in files:
                if fname.endswith('.txt'):
                    text = open(os.path.join(root, fname), 'rb').read()
                    yield text.decode('latin-1'), category

    df = pd.DataFrame(rows_generator(), columns=['text', 'category'])

    df.to_csv(output_file, index=False)


@classification.command()
@click.option('--input_data', default='data')
@click.option('--output_train', default='train.preprocessed.txt')
@click.option('--output_test', default='test.preprocessed.txt')
@click.option('--test_ratio', default=0.25)
def split(input_data, output_train, output_test, test_ratio):
    input_data_path = get_input_path(input_data)
    output_train_path = get_output_path(output_train)
    output_test_path = get_output_path(output_test)

    with open(input_data_path, 'r') as f:
        data = f.read().split('\n')

    # Shuffle data
    random.seed(RANDOM_SEED)
    random.shuffle(data)

    # Split train and test data
    index = round(len(data) * test_ratio)

    with open(output_test_path, 'w') as f:
        f.write('\n'.join(data[:index]))

    with open(output_train_path, 'w') as f:
        f.write('\n'.join(data[index:]))


@classification.command()
@click.option('--input_data', default='data')
@click.option('--output_data', default='data.preprocessed.txt')
def preprocess(input_data, output_data):
    input_data_path = get_input_path(input_data)
    output_data_path = get_output_path(output_data)

    df = pd.read_csv(
        input_data_path,
        engine='python')

    with open(output_data_path, 'w') as output:
        for f, c in zip(df[FEATURE_COLUMN], df[CATEGORY_COLUMN]):
            output.write(f'{process_string(f)} __label__{c}\n')


@classification.command()
@click.option('--input_train', default='train')
@click.option('--input_test', default='test')
@click.option('--output_model', default='train.model.bin')
@click.option('--output_parameters', default='parameters.json')
@click.option('--metric', default='f1')
@click.option('--k', default=1)
@click.option('--duration', default=300)
@click.option('--model_size', default='')
@click.option('--verbose', default=3)
def autotune(input_train, input_test, output_model, output_parameters,
    metric, k, duration, model_size, verbose):
    input_train_path = get_input_path(input_train)
    input_test_path = get_input_path(input_test)
    output_model_path = get_output_path(output_model)
    output_parameters_path = get_output_path(output_parameters)

    # Autotune model
    model = fasttext.train_supervised(
        input=input_train_path,
        autotuneValidationFile=input_test_path,
        autotuneMetric=metric,
        autotuneDuration=duration,
        autotuneModelSize=model_size,
        verbose=verbose)

    # Test best model
    n, p, r = model.test(input_test_path, k=k)
    print(json.dumps(
        {'n': n, 'precision': p, 'recall': r, 'k': k}))

    # Save best parameters
    best_parameters = get_model_parameters(model)
    print(json.dumps(best_parameters))
    with open(output_parameters_path, 'w') as f:
        json.dump(best_parameters, f)

    # Save best model
    model.save_model(output_model_path)


@classification.command()
@click.option('--input_data', default='data')
@click.option('--input_model', default='model')
@click.option('--output_predictions', default='predictions.csv')
@click.option('--k', default=1)
def predict(input_data, input_model, output_predictions, k):
    input_data_path = get_input_path(input_data)
    input_model_path = get_input_path(input_model)
    output_predictions_path = get_output_path(output_predictions)

    model = fasttext.load_model(input_model_path)

    # TODO: make this work also with unlabelled data
    with open(input_data_path) as f:
        df = pd.DataFrame(
            (split_text(line) for line in f),
            columns=['text', 'label'])

        all_labels, all_probs = model.predict(list(df['text']), k=k)

        columns = [f'prediction@{i}' for i in range(1, k+1)] + [f'p@{i}' for i in range(1, k+1)]
        predictions_df = pd.DataFrame((
            list(record_labels) + list(record_probs)
            for record_labels, record_probs in zip(all_labels, all_probs)
        ), columns=columns)

    df = df.join(predictions_df)

    # Remove __label__ from prediction columns
    for col in df.columns:
        if col.startswith('prediction@'):
            df[col] = df[col].str.replace('__label__', '')

    df.to_csv(output_predictions_path, index=False)


@classification.command()
@click.option('--input_data', default='data')
@click.option('--input_parameters', default='parameters')
@click.option('--output_model', default='data.model.bin')
def train(input_data, input_parameters, output_model):
    input_data_path = get_input_path(input_data)
    input_parameters_path = get_input_path(input_parameters)
    output_model_path = get_output_path(output_model)

    # Parse parameters
    with open(input_parameters_path) as f:
        parameters = json.load(f)

    # Train model
    model = fasttext.train_supervised(
        input=input_data_path,
        **parameters)

    # Save best model
    model.save_model(output_model_path)
