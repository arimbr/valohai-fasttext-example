import click

from models.classification.commands import classification


@click.group()
def cli():
    pass


cli.add_command(classification)

if __name__ == '__main__':
    cli()
