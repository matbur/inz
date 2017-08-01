class DNN:
    def __init__(self, network):
        self.network = network

    def fit(self, X_inputs: list, Y_targets: list, n_epoch=10, batch_size=64, shuffle=False):
        pass

    def get_weights(self):
        pass

    def load(self, model_file: str):
        pass

    def save(self, model_file: str):
        pass

    def predict(self, X: list):
        pass

    def predict_label(self, X: list):
        pass

    def set_weights(self, tensor, weights):
        pass


if __name__ == '__main__':
    x = []
    y = []
    net = None
    model = DNN(net)
    model.fit(x, y)
