from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

from inz import Model, fully_connected, input_data
from inz.utils import get_data, train_test_split, vector2onehot


def xor_problem():
    np.random.seed(1)
    x = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=float)
    y = np.array([
        [1, 0], [0, 1], [0, 1], [1, 0],
    ], dtype=float)

    net = input_data((None, 2))
    net = fully_connected(net, 3, activation='tanh')
    net = fully_connected(net, 2, activation='tanh')

    model = Model(net)
    model.fit(x, y, n_epoch=200)
    model.save('xor_model.json')
    model.load('xor_model.json')

    for i in zip(y, model.predict(x)):
        print(*i)

    model.plot_error()


def get_accuracy(pred, y):
    axis_ = np.argmax(pred, axis=1) - np.argmax(y, axis=1)
    return 1 - np.count_nonzero(axis_) / len(y)


def create_network(n, shapes, activation, seed=42):
    np.random.seed(seed)

    net = input_data(shape=(None, n))
    for i in shapes:
        net = fully_connected(net, i, activation)
    return net


def test_case(shapes, activation, n_features, batch_size, learning_rate, n_epoch, model_dir, seed=42):
    name = 's_{}_a_{}_f_{}_bs_{}_lr_{}'.format(
        '_'.join(map(str, shapes)), activation, n_features, batch_size, '_'.join(map(str, learning_rate))
    )
    print(name)
    network = create_network(n_features, shapes, activation, seed=seed)

    data = get_data(n_features)

    train, test = train_test_split(data, seed=seed)

    x_train = train[:, :-1]
    y_train = train[:, -1] - 1
    y_train = vector2onehot(y_train)
    x_test = test[:, :-1]
    y_test = test[:, -1] - 1
    y_test = vector2onehot(y_test)

    model = Model(network)
    model.fit(x_train, y_train,
              validation_set=(x_test, y_test),
              n_epoch=n_epoch,
              batch_size=batch_size,
              learning_rate=learning_rate,
              train_file=f'{model_dir}/{name}_train.json',
              )

    model_fn = f'{model_dir}/{name}_model.json'
    model.save(model_fn)
    # model.load(model_fn)

    # for i, j in zip(model.predict(x_test), y_test):
    #     print(np.argmax(i), np.argmax(j))

    print(get_accuracy(model.predict(x_test), y_test))

    # model.plot_error()


def wrapper(x):
    return test_case(**x)


def prepare_test_cases():
    model_dir = datetime.now().strftime('%s')
    Path(model_dir).mkdir(exist_ok=True)
    for act in ('sigmoid', 'tanh'):
        for feat in (10, 20, 30):
            for shape in ([24, 16, 12, 8], [16, 12, 8], [16, 8], [8]):
                for lr in ([.2, .2], [.2, .01], [.1, .1], [.1, .01], [.2, .001]):
                    yield {
                        'shapes': shape,
                        'activation': act,
                        'n_features': feat,
                        'batch_size': 10,
                        'learning_rate': lr,
                        'n_epoch': 400,
                        'model_dir': model_dir,
                    }


def run_all():
    np.random.seed(42)

    cpus = min(cpu_count(), 16)
    cases = prepare_test_cases()

    print(f'Running on {cpus} CPUs')

    with Pool(cpus) as pool:
        pool.map(wrapper, cases)


if __name__ == '__main__':
    from time import time

    t0 = time()
    run_all()
    # xor_problem()
    t = time() - t0
    print(f'Done in {t} s')
