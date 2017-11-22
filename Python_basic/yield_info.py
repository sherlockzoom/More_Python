# coding=utf-8
# 生成器使用 examples

def power(values):
    for value in values:
        print("powering %s " % value)
        yield value

def adders(values):
    for value in values:
        print("adding to %s "%value)
        if value%2==0:
            yield value+2
        else:
            yield value+3

elements = range(10)

results = adders(power(elements))

def psychologist():
    print("please tell me your problem")
    while True:
        answer = yield
        if answer is not None:
            if answer.endswith('?'):
                print("Don't ask me too much questions")
            elif "good" in answer:
                print("Ahh that's good")
            elif "bad" in answer:
                print("Don't be negative")

free = psychologist()
next(free)
free.send("ni hao ma?")
free.send("not good")
free.send("how bad day")
