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

    """ 
    This unexpectedly causes problems with training. Reason is unknown for now...
    Might be due to some numpy behavior. Multiplying by the inverse of the denominator 
    works for division by Non-Expression types, e.g. floats.
    
    def __truediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return Divide(self, other) """
        
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


class Variable(Expression):
    def __init__(self, value: float):
        self.value = value
        self.grad = 0.0

    def derive(self, seed: float):
        self.grad += seed


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


class Divide(Expression):
    def __init__(self, numerator: Expression, denominator: Expression):
        self.numerator = numerator
        self.denominator = denominator

    def eval(self):
        self.numerator.eval()
        self.denominator.eval()
        self.value = self.numerator.value / self.denominator.value

    def derive(self, seed: float):
        self.numerator.derive(1 / (self.denominator.value) * seed)
        self.denominator.derive(self.numerator.value / (self.denominator.value * self.denominator.value) * seed) 


class Abs(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def eval(self):
        self.x.eval()
        self.value = np.abs(self.x.value)
    
    def derive(self, seed: float):
        if self.x.value > 0:
            self.x.derive(seed)
        elif self.x.value < 0:
            self.x.derive(-seed)
        else:
            self.x.derive(0.0)


def absolute(x: Expression):
    return Abs(x)

def vec_abs(xs: list[Expression]):
    return [absolute(x) for x in xs]

def dot(a, b):
    assert len(a) == len(b)
    result = a[0] * b[0]
    for i in range(1, len(a)):
        result = result + (a[i] * b[i])
    return result



if __name__ == "__main__":
    x, y = Variable(2), Variable(3)

    z = dot([x, y], [y, x]) + 1

    z.eval()
    z.derive(1)

    print(x.grad)
    print(y.grad)
