import re
import json
import string
import io
import multiprocessing
import shutil

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
@click.option('--input_data', default='input_data')
@click.option('--output_preprocessed', default='preprocessed.txt')
@click.option('--feature', default='feature')
@click.option('--category', default='category')
@click.option('--separator', default=',')
def preprocess(input_data, output_preprocessed, feature, category, separator):
    input_data_path = get_input_path(input_data)
    output_preprocessed_path = get_output_path(output_preprocessed)

    df = pd.read_csv(
        input_data_path,
        sep=separator,
        engine='python')

    # TODO: operate on the DataFrame to add new column
    with open(output_preprocessed_path, 'w') as output:
        for f, c in zip(df[feature], df[category]):
            processed_f = process_string(f)
            output.write(f'{processed_f} __label__{c}\n')


@classification.command()
@click.option('--input_train', default='train')
@click.option('--input_test', default='test')
@click.option('--output_model', default='model.bin')
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
@click.option('--input_test', default='test')
@click.option('--input_model', default='model')
@click.option('--output_predictions', default='predictions.csv')
@click.option('--k', default=1)
def test(input_test, input_model, output_predictions, k):
    input_test_path = get_input_path(input_test)
    input_model_path = get_input_path(input_model)
    output_predictions_path = get_output_path(output_predictions)

    model = fasttext.load_model(input_model_path)

    with open(input_test_path) as f:
        df = pd.DataFrame(
            (split_text(line) for line in f),
            columns=['feature', 'label'])

        all_labels, all_probs = model.predict(list(df['feature']), k=k)

        columns = [f'prediction@{i}' for i in range(1, k+1)] + [f'p@{i}' for i in range(1, k+1)]
        predictions_df = pd.DataFrame((
            list(record_labels) + list(record_probs)
            for record_labels, record_probs in zip(all_labels, all_probs)
        ), columns=columns)

    df = df.join(predictions_df)

    # Remove __label__ from prediction columns
    for col in df.columns:
        if col.startswith('prediction@'):
            df[col] = df[col].str.lstrip('__label__')

    df.to_csv(output_predictions_path, index=False)


@classification.command()
@click.option('--input_train', default='train')
@click.option('--input_test', default='test')
@click.option('--input_parameters', default='parameters')
@click.option('--output_data', default='data.txt')
@click.option('--output_model', default='model.bin')
def train(input_train, input_test, input_parameters, output_data, output_model):
    input_train_path = get_input_path(input_train)
    input_test_path = get_input_path(input_test)
    input_parameters_path = get_input_path(input_parameters)
    output_data_path = get_output_path(output_data)
    output_model_path = get_output_path(output_model)

    # Concatenate train and test data
    with open(output_data_path, 'w') as output_data_file:
        for path in [input_train_path, input_test_path]:
            with open(path, 'r') as f:
                shutil.copyfileobj(f, output_data_file)

    # Parse parameters
    with open(input_parameters_path) as f:
        parameters = json.load(f)

    # Train model
    model = fasttext.train_supervised(
        input=output_data_path,
        **parameters)

    # Save best model
    model.save_model(output_model_path)
