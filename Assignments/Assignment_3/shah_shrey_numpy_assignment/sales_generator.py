"""generate a list of sales quantities with 10 rows and 5 columns."""

from random import randint


def main():
    print('[')
    for i in range(10):
        values = [randint(50, 2500) for i in range(5)]
        values_text = ', '.join(map(str, values))
        content = f'[{values_text}],'
        print(content)
    print(']')


main()
