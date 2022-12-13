import argparse

parser = argparse.ArgumentParser(description="Flornes porfolio selection model")
parser.add_argument('--queryID', '-id', type=str, help='Query ID')
parser.add_argument('--numLimit', '-n', type=int, default=5, help='Maximum number of constraints in each condition')
parser.add_argument('--threadLimit', '-t', type=int, default=4, help='Maximum number of threads')
args = parser.parse_args().__dict__
numLimit = args['numLimit']
threadLimit = args['threadLimit']
queryID = args['queryID']
print('Input argparse',  args)

if args['queryID']:
    print('Suceeded!')
else:
    print('Failure...')