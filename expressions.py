import numpy as np
import math


class Expression():

    def __init__(self):
        self.value = 0

    def eval(self):
        pass

    def derive(self, seed):
        pass

    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Plus(self, other)

    def __mul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Multiply(self, other)
        
    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Plus(self, Neg(other))

    def __neg__(self):
        return Neg(self)


class Constant(Expression):
    def __init__(self, value: float):
        self.value = value
        self.partial_derivative = 0.0

    def eval(self):
        pass

    def derive(self, seed: float):
        pass

class Variable(Expression):
    def __init__(self, value: float):
        self.value = value
        self.partial_derivative = 0.0

    def eval(self):
        pass

    def derive(self, seed: float):
        self.partial_derivative += seed


class Neg(Expression):
    def __init__(self, a):
        self.a = a

    def eval(self):
        self.a.eval()
        self.value = -self.a.value

    def derive(self, seed):
        self.a.derive(-seed)


class Plus(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b
    
    def eval(self):
        self.a.eval()
        self.b.eval()
        self.value = self.a.value + self.b.value

    def derive(self, seed: float):
        self.a.derive(seed)
        self.b.derive(seed)


class Multiply(Expression):
    def __init__(self, a: Expression, b: Expression):
        self.a = a
        self.b = b

    def eval(self):
        self.a.eval()
        self.b.eval()
        self.value = self.a.value * self.b.value
    
    def derive(self, seed: float):
        self.a.derive(self.b.value * seed)
        self.b.derive(self.a.value * seed)


class Activation(Expression):
    mode: str

    def __init__(self, x):
        pass


class ReLU(Activation):
    mode = 'scalar'

    def __init__(self, x: Expression):
        self.x = x

    def eval(self):
        self.x.eval()
        self.value = max(0, self.x.value)

    def derive(self, seed: float):
        if self.x.value > 0:
            self.x.derive(seed)


class Sigmoid(Activation):
    mode = 'scalar'

    def __init__(self, x: Expression):
        self.x = x

    def eval(self):
        self.x.eval()
        self.value = 1 / (1 + math.exp(-self.x.value))

    def derive(self, seed: float):
        self.x.derive(seed * self.value * (1 - self.value))


class Softmax(Activation):
    mode = "vector"
    
    def __init__(self, x_vec: list[Expression], index: int):
            self.x_vec = x_vec
            self.index = index
            self.value = None

    def eval(self):
        for xi in self.x_vec:
            xi.eval()
        exps = np.array([math.exp(xi.value) for xi in self.x_vec])
        self.value = exps[self.index] / exps.sum()

    def derive(self, seed):
        s = np.array([math.exp(xi.value) for xi in self.x_vec])
        s /= s.sum()
        grad = sum(seed * (s[j] * (1 if j==self.index else -s[self.index])) for j in range(len(s)))
        for xi in self.x_vec:
            xi.derive(grad)



def dot(a, b):
    assert len(a) == len(b)
    result = a[0] * b[0]
    for i in range(1, len(a)):
        result = result + (a[i] * b[i])
    return result

def wrap_into_variables(arr: np.ndarray):
    return np.vectorize(lambda x: Variable(x))(arr)

def map_to_value(arr: np.ndarray):
    return np.vectorize(lambda x: x.value)(arr)

vec_eval = np.vectorize(lambda x: x.eval())
vec_derive = lambda seed: np.vectorize(lambda x: x.derive(seed))


if __name__ == "__main__":
    x, y = Variable(2), Variable(3)

    z = dot([x, y], [y, x]) + 1

    z.eval()
    z.derive(1)

    print(x.partial_derivative)
    print(y.partial_derivative)
