import pickle

from Paths import seq1, out, seq2
from objectIO import seq2csv, seqs2csv

if __name__ == '__main__':
    seq2csv(seq1, out)
    seqs2csv([seq1, seq2], out, ['A', 'B'])
