import pickle
import argparse
import pprint

parser = argparse.ArgumentParser(description='''
Read and print data that was serialized by pickle.
''')

parser.add_argument('path')
args = parser.parse_args()

pp = pprint.PrettyPrinter()
print = pp.pprint

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def main():
    data = load_data(args.path)
    print(data)


if __name__ == '__main__':
    main()
