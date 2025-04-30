"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # TODO: (Extra Credit) You may set parameters for Adam optimizer here

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        # print(X.shape,W.shape)
        return X@W+b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> tuple:
        """Gradient of linear layer W@X+b
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
   
        de_dw = (X.T @ de_dz) 
        de_db = np.sum(de_dz, axis=0)
        de_dx = de_dz @ W.T  
        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0,X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return (X > 0).astype(float) 

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        posidx = x >= 0
        negidx = x < 0
        sig = np.zeros_like(x)
        
        sig[posidx] = 1 / (1 + np.exp(-x[posidx]))
        
        sig[negidx] = np.exp(x[negidx]) / (1 + np.exp(x[negidx]))
        return sig
        
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        sig = self.sigmoid(X)
        
        return (sig*(1-sig))+1e-8

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        n =y.shape[0]
        norm = np.sum((p - y) ** 2)/n
        return norm
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        n = y.shape[0]
        return (2*(p-y))/n
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # n = y.shape[0]
        return  (p - y) * p * (1 - p)          

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        self.X = X
        # for i in range(1,self.num_layers):    
        #     if i == 1:
        #         linout = self.linear(self.params["W"+str(i)],X,self.params["b"+str(i)])
        #     else:
        #         linout = self.linear(self.params["W"+str(i)],self.outputs["relu"+str(i-1)],self.params["b"+str(i)])
        #     self.outputs["relu"+str(i)] = self.relu(linout)
        # linout = self.linear(self.params["W"+str(self.num_layers)],self.outputs["relu"+str(self.num_layers-1)],self.params["b"+str(self.num_layers)])  
        # self.outputs["sigmoid"+str(self.num_layers)] = self.sigmoid(linout)
        # return self.outputs["sigmoid"+str(self.num_layers)]
        for i in range(1,self.num_layers):    
            if i == 1:
                self.outputs["linear"+str(i)] = self.linear(self.params["W"+str(i)],X,self.params["b"+str(i)])
            else:
                self.outputs["linear"+str(i)] = self.linear(self.params["W"+str(i)],self.outputs["relu"+str(i-1)],self.params["b"+str(i)])
            self.outputs["relu"+str(i)] = self.relu(self.outputs["linear"+str(i)])
        self.outputs["linear"+str(self.num_layers)] = self.linear(self.params["W"+str(self.num_layers)],self.outputs["relu"+str(self.num_layers-1)],self.params["b"+str(self.num_layers)])  
        self.outputs["sigmoid"+str(self.num_layers)] = self.sigmoid(self.outputs["linear"+str(self.num_layers)])
        return self.outputs["sigmoid"+str(self.num_layers)]


    def backward(self, y: np.ndarray) -> float:
    
        self.gradients = {}

        # Compute the loss
        
        m = 2.5
        finalout = self.outputs["sigmoid" + str(self.num_layers)]
        loss = self.mse(y, finalout)
        A_prev = self.outputs["relu" + str(self.num_layers-1)]
        W = self.params["W"+str(self.num_layers)]
        
        # print(mse_grad.shape,A_prev.shape) 
        dz = self.mse_sigmoid_grad(y,finalout)
      
        for i in range(self.num_layers, 0, -1):
            W = self.params["W" + str(i)]  
            A_prev = self.X if i == 1 else self.outputs["relu" + str(i - 1)]  

            dw, db, da = self.linear_grad(W, A_prev, dz)

            self.gradients["W" + str(i)] = dw / m
            self.gradients["b" + str(i)] = db / m

            if i > 1:
                dz = da * self.relu_grad(self.outputs["linear" + str(i - 1)]) 
         
        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for i in range(1, self.num_layers + 1):
                # print( self.params[f"b{i}"])
                self.params[f"W{i}"] -= lr * self.gradients[f"W{i}"]
                self.params[f"b{i}"] -= lr * np.squeeze(self.gradients[f"b{i}"])
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            pass
        else:
            raise NotImplementedError
        