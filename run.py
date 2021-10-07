import argparse

from dataset_class import *
import model as m

def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict next time order.'
    )
    parser.add_argument(
        'hidden_size',
        default=64,
        type=int,
        help='Number of sentences in a summary'
    )
    parser.add_argument(
        'batch_size',
        default=32,
        type=int,
        help='The size of the batch for training'
    )
    parser.add_argument(
        'learning_rate',
        default=0.01,
        type=float,
        help='Learning rate for optimization'
    )
    parser.add_argument(
        'num_epochs',
        default=30,
        type=int,
        help='Number of epochs for training'
    )
    parser.add_argument(
        'verbose',
        default=False,
        type=bool,
        help='Print state messages'
    )
    return parser.parse_args()

def save_prediction(prediction, name='submission'):
    name = ".".join((name, 'csv'))
    prediction.to_csv(name)


def transform_index(prediction, sample):
    prediction = pd.merge(
        sample,
        prediction,
        left_on='id',
        right_on='id',
        suffixes=("_x", "")
    )
    prediction = prediction.drop(['target_x'], axis=1)
    prediction = prediction.set_index('id')
    return prediction

def main():
    args = parse_args()
    history = pd.read_csv('train.csv')
    sample = pd.read_csv('sample_submission.csv')
    dataset = CartDataset.from_df(history)
    model = m.train_model(dataset, args)
    prediction = m.predict(dataset, model, args)
    prediction = transform_index(prediction, sample)
    save_prediction(prediction)

if __name__ == '__main__':
    main()