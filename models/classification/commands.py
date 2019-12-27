import re
import json
import string
import io
import multiprocessing

import click
import pandas as pd
import fasttext

from utils import get_input_path, get_output_path

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


def get_feature(text):
    return text.split('__label__')[0].strip()


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
@click.option('--input')
@click.option('--output')
@click.option('--id_vars')
@click.option('--value_vars')
@click.option('--var_name', default='category')
@click.option('--value_name', default='value')
@click.option('-s', '--separator', default=',')
def melt(input, output, id_vars, value_vars, var_name, value_name, separator):
    df = pd.read_csv(
        input,
        sep=separator,
        engine='python')
    df = pd.melt(
        df,
        id_vars=id_vars.split(','),
        value_vars=value_vars.split(','),
        var_name=var_name,
        value_name=value_name
    )

    # Drop not assigned features
    df[value_name] = df[value_name].fillna(0).astype(float)
    df = df[~df[value_name].isna() & (df[value_name] > 0)]

    df.to_csv(output, index=False)


@classification.command()
@click.option('--f1')
@click.option('--f2')
@click.option('--how', default='inner')
@click.option('--output')
@click.option('--separator', default=',')
def merge(f1, f2, how, output, separator):
    df1 = pd.read_csv(f1, sep=separator, engine='python')
    df2 = pd.read_csv(f2, sep=separator, engine='python')
    df = df1.merge(df2, how=how)

    df.to_csv(output, index=False)


@classification.command()
@click.option('--train', default='train')
@click.option('--train_preprocessed', default='train_preprocessed.txt')
@click.option('--feature', default='feature')
@click.option('--category', default='category')
@click.option('--separator', default=',')
def preprocess(train, train_preprocessed, feature, category, separator):
    train_path = get_input_path(train)
    train_preprocessed_path = get_output_path(train_preprocessed)

    df = pd.read_csv(
        train_path,
        sep=separator,
        engine='python')

    with open(train_preprocessed_path, 'w') as output:
        for f, c in zip(df[feature], df[category]):
            processed_f = process_string(f)
            output.write(f'{processed_f} __label__{c}\n')


@classification.command()
@click.option('--train_file', default='train')
@click.option('--test_file', default='test')
@click.option('--model_file', default='model.bin')
@click.option('--parameters_file', default='parameters.json')
@click.option('--metric', default='f1')
@click.option('--predictions', default=1)
@click.option('--duration', default=300)
@click.option('--model_size', default='')
@click.option('--verbose', default=3)
def autotune(train_file, test_file, model_file, parameters_file,
    metric, predictions, duration, model_size, verbose):
    train_file = get_input_path(train_file)
    test_file = get_input_path(test_file)
    model_file = get_output_path(model_file)
    parameters_file = get_output_path(parameters_file)

    # Train model
    model = fasttext.train_supervised(
        input=train_file,
        autotuneValidationFile=test_file,
        autotuneMetric=metric,
        autotuneDuration=duration,
        autotuneModelSize=model_size,
        verbose=verbose)

    # Save model
    model.save_model(model_file)

    # Save best parameters
    best_parameters = get_model_parameters(model)
    print(json.dumps(best_parameters))
    with open(parameters_file, 'w') as f:
        json.dump(best_parameters, f)


@classification.command()
@click.option('--test', default='test')
@click.option('--model', default='model')
@click.option('--k', default=1)
def test(test, model, k):
    test_path = get_input_path(test)
    model_path = get_input_path(model)

    model = fasttext.load_model(model_path)
    n, p, r = model.test(test_path, k=k)

    print(json.dumps(
        {'n': n, 'precision': p, 'recall': r, 'k': k}))


@classification.command()
@click.option('--test_file', default='test')
@click.option('--model_file', default='model')
@click.option('--predictions_file', default='predictions.csv')
@click.option('--k', default=1)
def predict(test_file, model_file, predictions_file, k):
    test_file = get_input_path(test_file)
    model_file = get_input_path(model_file)
    predictions_file = get_output_path(predictions_file)

    model = fasttext.load_model(model_file)

    with open(test_file) as f:
        labels, probas = model.predict(
            [get_feature(line) for line in f], k=k)

        df = pd.DataFrame(
            dict(zip(label, proba))
            for label, proba in zip(labels, probas)
        )
        df.columns = df.columns.str.lstrip('__label__')

    df.to_csv(predictions_file, index=False)
