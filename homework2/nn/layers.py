import numpy as np
from nn.init import initialize

class Layer:
    """Base class for all neural network modules.
    You must implement forward and backward method to inherit this class.
    All the trainable parameters have to be stored in params and grads to be
    handled by the optimizer.
    """
    def __init__(self):
        self.params, self.grads = dict(), dict()

    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *input):
        raise NotImplementedError


class Linear(Layer):
    """Linear (fully-connected) layer.

    Args:
        - in_dims (int): Input dimension of linear layer.
        - out_dims (int): Output dimension of linear layer.
        - init_mode (str): Weight initalize method. See `nn.init.py`.
          linear|normal|xavier|he are the possible options.
        - init_scale (float): Weight initalize scale for the normal init way.
          See `nn.init.py`.
        
    """
    def __init__(self, in_dims, out_dims, init_mode="linear", init_scale=1e-3):
        super().__init__()

        self.params["w"] = initialize((in_dims, out_dims), init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
    
    def forward(self, x):
        """Calculate forward propagation.

        Returns:
            - out (numpy.ndarray): Output feature of this layer.
        """
        ######################################################################
        # TODO: Linear 레이어의 forward propagation 구현.
        ######################################################################
        self.x_ = x
        out = np.dot(x, self.params["w"]) + self.params["b"]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        """Calculate backward propagation.

        Args:
            - dout (numpy.ndarray): Derivative of output `out` of this layer.
        
        Returns:
            - dx (numpy.ndarray): Derivative of input `x` of this layer.
        """
        ######################################################################
        # TODO: Linear 레이어의 backward propagation 구현.
        ######################################################################
        dx = np.dot(dout, self.params["w"].T)
        dw = np.dot(self.x_.T, dout)
        db = dout.sum(axis=0)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: ReLU 레이어의 forward propagation 구현.
        ######################################################################
        self.x_ = x
        out = x.copy()
        # out = x.copy()
        # for i, r in enumerate(x):
        #     for j, value in enumerate(r):
        #         out[i][j] = max(0, value)
        out = np.maximum(0, out)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: ReLU 레이어의 backward propagation 구현.
        ######################################################################
        x = self.x_.copy()
        neg = (x <= 0)
        dout[neg] = 0
        dx = dout
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Sigmoid 레이어의 forward propagation 구현.
        ######################################################################
        self.x_ = x.copy()
        out = 1 / (1+np.exp(-x))
        self.params["out"] = out
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Sigmoid 레이어의 backward propagation 구현.
        ######################################################################
        dx = dout * self.params["out"]*(1-self.params["out"])
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Tanh 레이어의 forward propagation 구현.
        ######################################################################
        self.x_ = x.copy()
        out = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        self.params["out"] = out
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Tanh 레이어의 backward propagation 구현.
        ######################################################################
        dx = dout * (1-self.params["out"])*(1+self.params["out"])
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class SoftmaxCELoss(Layer):
    """Softmax and cross-entropy loss layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """Calculate both forward and backward propagation.
        
        Args:
            - x (numpy.ndarray): Pre-softmax (score) matrix (or vector).
            - y (numpy.ndarray): Label of the current data feature.

        Returns:
            - loss (float): Loss of current data.
            - dx (numpy.ndarray): Derivative of pre-softmax matrix (or vector).
        """
        ######################################################################
        # TODO: Softmax cross-entropy 레이어의 구현. 
        #        
        # NOTE: 이 메소드에서 forward/backward를 모두 수행하고, loss와 gradient (dx)를 
        # 리턴해야 함.
        ######################################################################
        y_pred = []
        for i in range(len(y)):
            y_pred.append(np.exp(x[i]) / np.sum(np.exp(x[i])))
        y_pred = np.array(y_pred)
        y_log = []
        for i in range(len(y)):
            y_log.append(y_pred[i][y[i]])
        loss = np.sum(-np.log(np.array(y_log)))/len(y)

        for i in range(len(y)):
            y_pred[i][y[i]] -= 1
        dx = y_pred/len(y)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss, dx
    
    
class Conv2d(Layer):
    """Convolution layer.

    Args:
        - in_dims (int): Input dimension of conv layer.
        - out_dims (int): Output dimension of conv layer.
        - ksize (int): Kernel size of conv layer.
        - stride (int): Stride of conv layer.
        - pad (int): Number of padding of conv layer.
        - Other arguments are same as the Linear class.
    """
    def __init__(
        self,
        in_dims, out_dims,
        ksize, stride, pad,
        init_mode="linear",
        init_scale=1e-3
    ):
        super().__init__()
        
        self.params["w"] = initialize(
            (out_dims, in_dims, ksize, ksize), 
            init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        ######################################################################
        # TODO: Convolution 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        self.x_ = x.copy()
        
        strd = self.stride

        x_pad = np.pad(x,((0,),(0,), (self.pad,), (self.pad,)), 'constant')
        NX, HX, RX, CX = x_pad.shape
        NW, HW, RW, CW = self.params["w"].shape

        oh = int((RX-RW)/strd+1)
        ow = int((CX-CW)/strd+1)

        out = np.zeros((NX, NW, oh, ow))
        for ipt in range(NX):
            for f in range(NW):
                for i in range(int((RX-RW)/strd+1)):
                    for j in range(int((CX-CW)/strd+1)):
                        out[ipt,f, i, j] = np.sum(x_pad[ipt][:, i*strd:i*strd+RW, j*strd:j*strd+CW] * self.params["w"][f])
                        # out.append(np.sum(x_pad[ipt][:, i*strd:i*strd+RW, j*strd:j*strd+CW] * self.params["w"][f]))
        # outnum = NX
        # fn = NW
        # oh = int((RX-RW)/strd+1)
        # ow = int((CX-CW)/strd+1)

        # out = np.array(out).reshape((outnum, fn, oh, ow))
        out += np.reshape(self.params["b"], (len(self.params["b"]),1,1))

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Convolution 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        x = self.x_.copy()
        strd = self.stride

        x_pad = np.pad(x,((0,),(0,), (self.pad,), (self.pad,)), 'constant')
        NX, HX, RX, CX = x_pad.shape
        NW, HW, RW, CW = self.params["w"].shape
        dx_pad = np.zeros(x_pad.shape)
        dw = np.zeros((NW, HW, RW, CW))
        for ipt in range(NX):
            for f in range(NW):
                for i in range(int((RX-RW)/strd+1)):
                    for j in range(int((CX-CW)/strd+1)):
                        dx_pad[ipt][:,i*strd:i*strd+RW,j*strd:j*strd+CW]+=self.params["w"][f]*dout[ipt,f,i,j]
                        dw[f]+=x_pad[ipt][:,i*strd:i*strd+RW,j*strd:j*strd+CW]*dout[ipt,f,i,j]
        
        dx = dx_pad[:,:,self.pad:RX-self.pad, self.pad:CX-self.pad]
        db = []
        for b_num in range(NW):
            db.append(np.sum(dout[:, b_num, :, : ]))
        db = np.array(db)
        db.reshape(self.params["b"].shape)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx
    

class MaxPool2d(Layer):
    """Max pooling layer.

    Args:
        - ksize (int): Kernel size of maxpool layer.
        - stride (int): Stride of maxpool layer.
    """
    def __init__(self, ksize, stride):
        super().__init__()
        
        self.ksize = ksize
        self.stride = stride
        
    def forward(self, x):
        ######################################################################
        # TODO: Max pooling 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 2-중첩으로 구현.
        ######################################################################
        self.x_ = x.copy()
        strd = self.stride
        ksize = self.ksize
        N, H, R, C = x.shape
        out = np.zeros((N, H, int((R-ksize)/strd+1), int((C-ksize)/strd+1)))
        for i in range(int((R-ksize)/strd+1)):
            for j in range(int((C-ksize)/strd+1)):
                out[:,:,i,j] = np.max(x[:,:,i*strd:i*strd+ksize, j*strd:j*strd+ksize], axis=(2,3))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Max pooling 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        x = self.x_.copy()
        strd = self.stride
        ksize = self.ksize
        N, H, R, C = x.shape
        dx = np.zeros(x.shape)
        for ipt in range(N):
            for h in range(H):
                for i in range(int((R-ksize)/strd+1)):
                    for j in range(int((C-ksize)/strd+1)):
                        mx = np.argmax(x[ipt,h,i*strd:i*strd+ksize, j*strd:j*strd+ksize])
                        dx[ipt,h,i*strd+mx//ksize,j*strd+mx%ksize]=dout[ipt,h,i,j]

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx
