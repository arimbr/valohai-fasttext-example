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


TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
LABEL_SEPARATOR = '__label__'
PROBABILITY_COLUMN = 'p'
RANDOM_SEED = 42
VERBOSE = 3

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
    'label': LABEL_SEPARATOR,
    'verbose': 2,
    'pretrainedVectors': '',
    'seed': 0,
}


CLEAN_LABEL_REGEX = re.compile(r'{}'.format(LABEL_SEPARATOR))


def format_label(label):
    return re.sub(CLEAN_LABEL_REGEX, '', label)


def format_labels(labels):
    return [format_label(label) for label in labels]


def not_empty_str(x):
    return isinstance(x, str) and x != ''


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
    text, label = text.split(LABEL_SEPARATOR)
    return text.strip(), label.strip()


def process_text(text):
    # Transform multiple spaces and \n to a single space
    text = re.sub(r'\s{1,}', ' ', text)
    # Remove punctuation
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    text = text.translate(remove_punct_map)
    # Transform to lowercase
    text = text.lower()
    return text


def get_predictions_df(all_labels, all_probs, k):
    labels_columns = [f'{LABEL_COLUMN}@{i}' for i in range(1, k+1)]
    probs_columns = [f'{PROBABILITY_COLUMN}@{i}' for i in range(1, k+1)]

    return pd.DataFrame((
        format_labels(labels) + list(probs)
        for labels, probs in zip(all_labels, all_probs)
    ), columns=labels_columns + probs_columns)


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

    df = pd.DataFrame(rows_generator(), columns=[TEXT_COLUMN, LABEL_COLUMN])

    df.to_csv(output_file, index=False)


@classification.command()
@click.option('--input_data', default='data')
@click.option('--output_data', default='preprocessed.txt')
@click.option('--text_column', default=TEXT_COLUMN)
@click.option('--label_column', default=LABEL_COLUMN)
def preprocess(input_data, output_data, text_column, label_column):
    # TODO: make it work also with prediction data without label
    input_data_path = get_input_path(input_data)
    output_data_path = get_output_path(output_data)

    df = pd.read_csv(
        input_data_path,
        engine='python').fillna('')

    # Concatenate strings if multiple text columns
    if ',' in text_column:
        df[text_column] = df[text_column.split(',')].agg(' '.join, axis=1)

    with open(output_data_path, 'w') as output:
        for text, label in zip(df[text_column], df[label_column]):
            if not_empty_str(text) and not_empty_str(label):
                output.write(f'{process_text(text)} {LABEL_SEPARATOR}{label}\n')


@classification.command()
@click.option('--input_data', default='data')
@click.option('--output_train', default='train.txt')
@click.option('--output_validation', default='validation.txt')
@click.option('--output_test', default='test.txt')
@click.option('--train_ratio', default=0.8)
@click.option('--validation_ratio', default=0.1)
@click.option('--test_ratio', default=0.1)
@click.option('--shuffle', is_flag=True)
def split(input_data, output_train, output_validation, output_test,
    train_ratio, validation_ratio, test_ratio, shuffle):
    input_data_path = get_input_path(input_data)
    output_train_path = get_output_path(output_train)
    output_validation_path = get_output_path(output_validation)
    output_test_path = get_output_path(output_test)

    with open(input_data_path, 'r') as f:
        data = f.read().strip().split('\n')

    # Shuffle data
    if shuffle:
        print('Shuffling data')
        random.seed(RANDOM_SEED)
        random.shuffle(data)

    # Split train, validation and test data
    validation_index = round(len(data) * train_ratio)
    test_index = round(len(data) * (train_ratio + validation_ratio))
    end_index = round(len(data) * (train_ratio + validation_ratio + test_ratio))

    with open(output_train_path, 'w') as f:
        f.write('\n'.join(data[:validation_index]))

    with open(output_validation_path, 'w') as f:
        f.write('\n'.join(data[validation_index:test_index]))

    with open(output_test_path, 'w') as f:
        f.write('\n'.join(data[test_index:end_index]))


@classification.command()
@click.option('--input_train', default='train')
@click.option('--input_validation', default='validation')
@click.option('--output_model', default='train_model.bin')
@click.option('--output_parameters', default='parameters.json')
@click.option('--metric', default='f1')
@click.option('--k', default=1)
@click.option('--duration', default=1200)
@click.option('--model_size', default='2000M')
def autotune(input_train, input_validation, output_model, output_parameters,
    metric, k, duration, model_size):
    input_train_path = get_input_path(input_train)
    input_validation_path = get_input_path(input_validation)
    output_model_path = get_output_path(output_model)
    output_parameters_path = get_output_path(output_parameters)

    # Autotune model
    model = fasttext.train_supervised(
        input=input_train_path,
        autotuneValidationFile=input_validation_path,
        autotuneMetric=metric,
        autotuneDuration=duration,
        autotuneModelSize=model_size,
        verbose=VERBOSE)

    # Log best model metrics
    n, p, r = model.test(input_validation_path, k=k)
    print(json.dumps(
        {'n': n, 'precision': p, 'recall': r, 'k': k}))

    # Save best parameters
    with open(output_parameters_path, 'w') as f:
        json.dump(get_model_parameters(model), f)

    # Save best model
    model.save_model(output_model_path)


@classification.command()
@click.option('--input_data', default='data')
@click.option('--input_parameters', default='parameters')
@click.option('--output_model', default='model.bin')
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

    # Save model
    model.save_model(output_model_path)


@classification.command()
@click.option('--input_test', default='test')
@click.option('--input_model', default='model')
@click.option('--output_predictions', default='test_predictions.csv')
@click.option('--k', default=1)
def test(input_test, input_model, output_predictions, k):
    input_test_path = get_input_path(input_test)
    input_model_path = get_input_path(input_model)
    output_predictions_path = get_output_path(output_predictions)

    model = fasttext.load_model(input_model_path)

    # Log model metrics
    n, p, r = model.test(input_test_path, k=k)
    print(json.dumps(
        {'n': n, 'precision': p, 'recall': r, 'k': k}))

    # Split feature and category in a DataFrame
    with open(input_test_path) as f:
        df = pd.DataFrame(
            (split_text(line) for line in f),
            columns=[TEXT_COLUMN, LABEL_COLUMN])

    # Get predictions
    all_labels, all_probs = model.predict(
        list(df[TEXT_COLUMN]), k=k)

    # Add formatted predictions
    predictions_df = get_predictions_df(all_labels, all_probs, k)
    df = df.join(predictions_df)

    # Add error column
    df['error'] = (df[f'{LABEL_COLUMN}'] != df[f'{LABEL_COLUMN}@1'])

    # Save predictions
    df.to_csv(output_predictions_path, index=False)


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

    # Create text DataFrame
    with open(input_data_path) as f:
        df = pd.DataFrame(
            (line for line in f),
            columns=[TEXT_COLUMN])

    # Get predictions
    all_labels, all_probs = model.predict(
        list(df[TEXT_COLUMN]), k=k)

    # Add formatted predictions
    predictions_df = get_predictions_df(all_labels, all_probs, k)
    df = df.join(predictions_df)

    # Save predictions
    df.to_csv(output_predictions_path, index=False)
