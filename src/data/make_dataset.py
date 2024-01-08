# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('processing pistachio images')

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std as needed
    ])

    # Load and preprocess images from both classes
    dataset = datasets.ImageFolder(root=input_filepath, transform=transform)

    # Combine images and labels into a single tensor
    data_tensor = torch.stack([img for img, _ in dataset])
    labels_tensor = torch.tensor([label for _, label in dataset])

    # Save the intermediate representation
    processed_data = {'data': data_tensor, 'labels': labels_tensor}
    output_filepath = Path(output_filepath)
    torch.save(processed_data, output_filepath)

    logger.info('Processing completed. Intermediate representation saved to: %s', output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
