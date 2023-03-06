import numpy as np


def extract_val(x):
    if isinstance(x, Iteration):
        return x.value
    else:
        return x


class Iteration:
    """ This class keeps track of iterations across the python program. Iteration numbers are stored centrally and only
    once. Access is done via creation of this object with a key, and each key has an unique iteration counter attached
    to it.

    By initializing an Iteration object with a name, a new counter is created by default on zero
    >>> it = Iteration('my_counter')
    >>> it.value
    0

    Incrementation can be done by the method incr()
    >>> it.incr()
    1

    Normal comparisons can be used
    >>> it == 1
    True
    >>> it > 3
    False
    >>> it != 2
    True

    Just like normal math operations (only in-place)
    >>> it += 4
    >>> it.value
    5
    >>> it % 2
    1
    >>> it // 2
    2

    Iterators with different names can be stored in parallel
    >>> it1 = Iteration('other_counter', 3)
    >>> it1.value
    3
    >>> it.value
    5

    And new iterators can be created using old names
    >>> it2 = Iteration('my_counter')
    >>> it2 == it == 5
    True

    To reset (or not initialize to 0) a counter, you can use
    >>> it3 = Iteration('my_counter', 0)
    >>> it == it2 == it3 == 0
    True

    THE ONLY THING YOU SHOULDN'T DO IS REASSIGN WITH AN INTEGER. IT STOPS BEING AN ITERATION OBJECT
    >>> it = 8
    >>> it == 8
    True
    >>> it2 == it
    False

    """
    storage: dict = {}
    default_start: int = 0

    def __init__(self, key: str, start: int = None):
        self.key = key

        intval = Iteration.default_start if start is None else start
        is_created = self.key in Iteration.storage.keys()

        if not is_created or start is not None:
            Iteration.storage[self.key] = intval

    @property
    def value(self):
        return Iteration.storage[self.key]

    @value.setter
    def value(self, v):
        Iteration.storage[self.key] = v

    # Math operators
    def incr(self):
        self.__iadd__(1)
        return self

    def decr(self):
        self.__isub__(1)
        return self

    def __iadd__(self, value: int):
        self.value += value
        return self

    def __isub__(self, value: int):
        self.value -= value
        return self

    def __imul__(self, value: int):
        self.value *= int(value)
        return self

    def __idiv__(self, value):
        self.value = int(self.value / value)
        print("warning: Iteration divided -> parse to an int()")
        return self

    def __mod__(self, value):
        return self.value.__mod__(value)

    def __divmod__(self, value):
        return self.value.__divmod__(value)

    def __abs__(self):
        return self.value.__abs__()

    def __ceil__(self):
        return self.value.__ceil__()

    def __floor__(self):
        return self.value.__floor__()

    def __floordiv__(self, value):
        return self.value.__floordiv__(value)

    def __invert__(self):
        return self.value.__invert__()

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __lshift__(self, value):
        return self.value.__lshift__(value)

    def __rshift__(self, value):
        return self.value.__rshift__(value)

    def __rlshift__(self, value):
        return self.value.__rlshift__(value)

    # Comparisons
    def __bool__(self):
        return self.value.__bool__()

    def __and__(self, value):
        return self.value.__and__(extract_val(value))

    def __or__(self, value):
        return self.value.__or__(extract_val(value))

    def __eq__(self, value):
        return self.value == extract_val(value)

    def __ne__(self, value):
        return self.value != extract_val(value)

    def __ge__(self, value):
        return self.value >= extract_val(value)

    def __gt__(self, value):
        return self.value > extract_val(value)

    def __le__(self, value):
        return self.value <= extract_val(value)

    def __lt__(self, value):
        return self.value < extract_val(value)

    # Conversion
    def __float__(self):
        return self.value.__float__()

    def __int__(self):
        return self.value.__int__()

    # Representations
    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__str__()

    def __format__(self, format_spec):
        return self.value.__format__(format_spec)


class Parameter:
    """ This class behaves just like a value, but can change over time
    >>> v = Parameter(3)
    >>> print(v * 3)
    9
    >>> print(3*v)
    9
    """

    def __init__(self, val0=-np.inf, val1=np.inf):
        self.minval = min(val0, val1)
        self.maxval = max(val0, val1)

    def calculate_value(self):
        return 1.0

    @property
    def value(self):
        return np.clip(self.calculate_value(), self.minval, self.maxval)

    # Mathematical operators
    def __add__(self, other):
        return self.value+other

    def __radd__(self, other):
        return other+self.value

    def __sub__(self, other):
        return self.value-other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value*other

    def __rmul__(self, other):
        return other*self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    # Conversion
    def __float__(self):
        return self.value.__float__()

    def __int__(self):
        return self.value.__int__()

    # Representations
    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__str__()

    def __format__(self, format_spec):
        return self.value.__format__(format_spec)


class Linear(Parameter):
    """ aI + b"""
    def __init__(self, startval: float, endval: float, istart: int, iend: int, it: Iteration, interval: int = 1):
        """
        :param startval: Starting value
        :param endval: End value
        :param istart: Iteration to start increasing
        :param iend: Iteration to reach the ending value
        :param it: Iteration counter
        :param interval: Update every Nth iteration
        """
        super().__init__(startval, endval)
        self.it = it
        self.interval = interval
        self.a = (endval - startval) / (iend - istart)
        self.b = startval - self.a * istart

    def calculate_value(self):
        return self.a * (int(self.it)//self.interval)*self.interval + self.b


class Exponential(Parameter):
    """ exp(aI + b)"""
    def __init__(self, startval: float, endval: float, istart: int, iend: int, it: Iteration):
        super().__init__(startval, endval)
        self.it = it
        self.b = (np.log(self.minval)*iend - np.log(self.maxval)*istart) / (iend - istart)
        self.a = (np.log(self.maxval) - np.log(self.minval)) / (iend - istart)

    def calculate_value(self):
        return np.exp(self.a * int(self.it) + self.b)


class Step(Parameter):
    def __init__(self, vals, it: Iteration):
        """
        :param vals: [(0, 0.0), (10, 1.0), (200, 5.0)] -> 0.0 for [0 ... 9], 1.0 for [10 ... 199], 5.0 for [200 ... )
        :param it:
        """
        self.it_list = np.array([v[0] for v in vals])
        self.val_list = np.array([v[1] for v in vals])
        self.it = it
        assert np.all(np.diff(self.it_list) > 0)
        super().__init__()

    def calculate_value(self):
        ipos = np.argwhere(int(self.it) < self.it_list).flatten()
        if len(ipos) == 0:
            i = -1
        else:
            i = ipos[0]-1
        if isinstance(self.val_list[i], Parameter):
            return self.val_list[i].value
        else:
            return self.val_list[i]


class DoubleEveryN(Parameter):
    def __init__(self, startval, endval, itstart, interval, it: Iteration):
        super().__init__(startval, endval)
        self.it = it
        self.itstart = itstart
        self.interval = interval
        self.startval = startval

    def calculate_value(self):
        return self.startval * 2**((int(self.it)-self.itstart)//self.interval)


if __name__ == "__main__":
    v = Parameter(5)

    print((v/np.ones(3)).dtype)

    print((np.ones(3)/float(v)).dtype)
    exit()

    print("-- Create a")
    a = Iteration('mma', 0)
    print("a = ", a)

    print("-- Create b")
    b = Iteration('mmasub', 5)
    print("a = ", a)
    print("b = ", b)

    print("-- Copy a to c")
    c = a
    print("a = ", a)
    print("b = ", b)
    print("c = ", c)

    print("-- Copy b to d")
    d = b
    print("a, b, c, d = ", a, b, c, d)

    print("-- Increment a")
    a.incr()
    print("a, b, c, d = ", a, b, c, d)

    print("-- Increment b")
    b.incr()
    print("a, b, c, d = ", a, b, c, d)

    print("-- Increment c")
    c.incr()
    print("a, b, c, d = ", a, b, c, d)

    print("-- Increment d")
    d.incr()
    print("a, b, c, d = ", a, b, c, d)

    print("-- Format a")
    print("{0: 4d}".format(a))

    print("-- Increment general mma (= a & c)")
    Iteration('mma').incr()
    print("a, b, c, d = ", a, b, c, d)

    print("-- Reset mma counter (= a & c)")
    Iteration('mma', 0)
    print("a, b, c, d = ", a, b, c, d)

    print("-- a == 0 = True: ", a == 0)
    print(a < 3.5)
    print(a < 1)

    a += 3
    print(a)
    print(type(a))

    import doctest
    doctest.testmod()