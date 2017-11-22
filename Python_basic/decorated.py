# coding=utf-8 

# 1 装饰器基本语法, 语法糖

@some_decorator
def decorated_function():
    pass

def decorated_function():
    pass
# 1.1 装饰器的等价用法
decorated_function = some_decorator(decorated_function)

# 2 装饰函数

def mydecorator(function):
    def wrapped(*args, **kwargs):
        # do something else
        result = function(*args, **kwargs)
        # 在函数调用之后，做点什么
        # 返回结果
        return result
    # 返回wrapper作为装饰函数
    return wrapped

# 3. 类装饰器

class DecoratorAsClass:
    def __init__(self, function):
        self.function = function 

    def __call__(self, *args, **kwargs):
        # 在调用之前，做点什么
        result = self.function(*args, **kwargs)
        # 在调用之后，做点什么
        # 返回结果
        return result

# 4. 参数化装饰器

def repeat(number=3):
    """多次重复执行装饰函数

    返回最后一次原始函数调用作为结果
    :param number: 重复次数，默认值是3
    """
    def actual_decorator(function):
        def wrapper(*args, **kwargs):
            result = None 
            for _ in range(number):
                result = function(*args, **kwargs)
            return result
        return wrapper
    return actual_decorator

@repeat(4)
def foo():
    print("call foo")
