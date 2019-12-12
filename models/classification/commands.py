import json
import string

import click
import pandas as pd
import fasttext

from utils import get_input_path, get_output_path


def process_string(s):
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    s = s.translate(remove_punct_map)
    s = s.lower()
    return s


@click.group()
def classification():
    pass


@classification.command()
@click.option('-i', '--input_name')
@click.option('-o', '--output_file')
@click.option('-f', '--feature', default='feature')
@click.option('-c', '--category', default='category')
@click.option('-s', '--separator', default=',')
def preprocess(input_name, output_file, feature, category, separator):
    input_path = get_input_path(input_name)
    output_path = get_output_path(output_file)

    df = pd.read_csv(
        input_path,
        sep=separator,
        engine='python')

    with open(output_path, 'w') as output:
        for f, c in zip(df[feature], df[category]):
            processed_f = process_string(f)
            output.write(f'{processed_f} __label__{c}\n')


@classification.command()
@click.option('-i', '--input_name')
@click.option('-o', '--output_file')
@click.option('--lr', default=0.01)
@click.option('--minCount', default=2)
def train(input_name, output_file, lr, mincount):
    input_path = get_input_path(input_name)
    output_path = get_output_path(output_file)

    model = fasttext.train_supervised(
        input=input_path, lr=lr, minCount=mincount)
    model.save_model(output_path)


@classification.command()
@click.option('-i', '--input_name')
@click.option('-m', '--model_name')
@click.option('--k', default=1)
def test(input_name, model_name, k):
    input_path = get_input_path(input_name)
    model_path = get_input_path(model_name)

    model = fasttext.load_model(model_path)
    n, p, r = model.test(input_path, k=k)

    print(json.dumps(
        {'n': n, 'precision': p, 'recall': r, 'k': k}))
