import re
import json
import string

import click
import pandas as pd
import fasttext

from utils import get_input_path, get_output_path


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
@click.option('--train', default='train_preprocessed')
@click.option('--model', default='model.bin')
@click.option('--lr', default=0.01)
@click.option('--minCount', default=2)
def train(train, model, lr, mincount):
    train_path = get_input_path(train)
    model_path = get_output_path(model)

    model = fasttext.train_supervised(
        input=train_path, lr=lr, minCount=mincount)
    model.save_model(model_path)


@classification.command()
@click.option('--train', default='train_preprocessed')
@click.option('--test', default='test_preprocessed')
@click.option('--model', default='model.bin')
@click.option('--autotuneMetric', default='f1')
@click.option('--autotuneDuration', default=60*5)
@click.option('--verbose', default=3)
def tune(train, test, model, autotunemetric, autotuneduration, verbose):
    train_path = get_input_path(train)
    test_path = get_input_path(test)
    model_path = get_output_path(model)

    model = fasttext.train_supervised(
        input=train_path,
        autotuneValidationFile=test_path,
        autotuneMetric=autotunemetric,
        autotuneDuration=autotuneduration,
        verbose=verbose)

    model.save_model(model_path)


@classification.command()
@click.option('--test', default='test_preprocessed')
@click.option('--model', default='model')
@click.option('--k', default=1)
def test(test, model, k):
    test_path = get_input_path(test)
    model_path = get_input_path(model)

    model = fasttext.load_model(model_path)
    n, p, r = model.test(test_path, k=k)

    print(json.dumps(
        {'n': n, 'precision': p, 'recall': r, 'k': k}))
