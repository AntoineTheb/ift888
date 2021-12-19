import numpy as np


class TwoLayerClassifier(object):
    def __init__(self, X_train, y_train, X_val, y_val, loss, num_features,
                 num_hidden_neurons, num_classes, activation='relu'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.num_features = num_features
        self.num_classes = num_classes

        self.net = TwoLayerNet(
            num_features, num_hidden_neurons, num_classes, loss, activation)

        self.momentum_cache_v_prev = {}

    def train(
        self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, momentum=0.0
    ):
        loss_train_curve = []
        loss_val_curve = []
        accu_train_curve = []
        accu_val_curve = []

        self.net.reinit()
        self.net.l2_reg = l2_reg

        self.momentum_cache_v_prev = {
            id(x): np.zeros_like(x) for x in self.net.parameters}

        sample_idx = 0
        num_iter = num_epochs * len(self.X_train)
        for i in range(num_iter):
            # Take a sample
            X_sample = self.X_train[sample_idx]
            y_sample = self.y_train[sample_idx]

            # Forward + Backward
            loss_train = self.net.forward_backward(X_sample, y_sample)

            # Take gradient step
            for w, dw in zip(self.net.parameters, self.net.gradients):
                self.momentum_update(w, dw, lr, momentum)

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.X_train):  # End of epoch

                accu_train, loss_train = \
                    self.global_accuracy_and_loss(
                        self.X_train, self.y_train)
                accu_val, loss_val, = \
                    self.global_accuracy_and_loss(
                        self.X_val, self.y_val)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0

                lr *= lr_decay

        return (loss_train_curve, loss_val_curve,
                accu_train_curve, accu_val_curve)

    def predict(self, x):
        if len(x.shape) == 1:  # Predict on one sample
            scores = self.net.forward(x)
            return np.argmax(scores, axis=1)
        elif len(x.shape) == 2:  # Predict on multiple samples
            pred = []
            for xs in x:
                pred.append(self.net.forward(xs))
            return np.argmax(pred, axis=1)

    def global_accuracy_and_loss(self, X, y, l2_r=-1.0):
        # TODO regularization
        """
        Compute average accuracy and loss for a series of
        N data points. Naive implementation (with loop)
        Inputs:
        - X: A numpy array of shape (D, N) containing a sample.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        if l2_r > 0:
            self.net.l2_reg = l2_r
        #######################################################################
        # TODO: Compute the softmax loss & accuracy for a series of samples X,y
        #######################################################################
        loss = 0
        for xs, ys in zip(X, y):
            scores = self.net.forward(xs)
            loss_s, _ = self.net.loss(scores, ys)
            loss += loss_s
        loss /= len(X)

        pred = self.predict(X)
        accu = (pred == y).mean()
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return accu, loss

    def momentum_update(self, w, dw, lr, mu):
        v_prev = self.momentum_cache_v_prev[id(w)]
        v = mu * v_prev - lr * dw  # integrate velocity
        w += v  # integrate position
        self.momentum_cache_v_prev[id(w)] = v


class TwoLayerNet(object):
    def __init__(
        self, in_size, hidden_size, num_classes, loss, activation='relu', l2_r=0.0
    ):
        self.in_size = in_size
        self.num_classes = num_classes
        self.l2_reg = l2_r
        TwoLayerNet.loss = loss
        self.layer1 = DenseLayer(in_size, hidden_size, activation=activation)
        self.layer2 = DenseLayer(hidden_size, num_classes)

    def reinit(self):
        self.layer1.reinit()
        self.layer2.reinit()

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x

    def backward_(self, dloss_dscores):
        dx = self.layer2.backward(dloss_dscores, self.l2_reg)
        self.layer1.backward(dx, self.l2_reg)

    def forward_backward(self, x, y):
        self.layer1.zero_grad()
        self.layer2.zero_grad()

        scores = self.forward(x)
        loss, dscores = self.loss(scores, y)
        self.backward_(dscores)
        return loss

    @property
    def parameters(self):
        return [self.layer1.W, self.layer2.W]

    @property
    def gradients(self):
        return [self.layer1.dW, self.layer2.dW]


class DenseLayer(object):
    def __init__(self, in_size, out_size, activation=None):
        self.activation = activation
        self.W = None
        self.dW = None
        self.in_size = in_size
        self.out_size = out_size
        self.reinit()

        self.last_x = None
        self.last_activ = None

    def reinit(self):
        self.W = np.random.randn(
            self.in_size + 1, self.out_size) * \
            np.sqrt(0.1 / (self.in_size + self.out_size))
        self.dW = np.zeros_like(self.W)

    def zero_grad(self):
        self.dW.fill(0.0)

    def forward(self, x):
        x = augment(x)
        f = x.dot(self.W)  # class scores
        if self.activation == 'sigmoid':
            f = sigmoid(f)
        elif self.activation == 'relu':
            f = np.maximum(0, f)
        self.last_x = x
        self.last_activ = f
        return f

    def backward(self, dnext_dout, l2_reg):
        if self.activation == 'sigmoid':
            dout_dsigmoid = self.last_activ * (1.0 - self.last_activ)
            dnext_dsigmoid = dnext_dout * dout_dsigmoid  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * \
                dnext_dsigmoid[np.newaxis, :]
            dnext_dX = dnext_dsigmoid.dot(self.W.T)
        elif self.activation == 'relu':
            dout_dsigmoid = self.last_activ != 0.0
            dnext_dsigmoid = dnext_dout * dout_dsigmoid  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * \
                dnext_dsigmoid[np.newaxis, :]
            dnext_dX = dnext_dsigmoid.dot(self.W.T)
        else:
            dnext_dW = self.last_x[:, np.newaxis].dot(
                dnext_dout[np.newaxis, :])
            dnext_dX = dnext_dout.dot(self.W.T)
        # discard the gradient wrt the 1.0 of homogeneous coord
        dnext_dX = dnext_dX[:-1]

        self.dW += dnext_dW
        self.dW += l2_reg * self.W  # add regul. gradient
        return dnext_dX


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
