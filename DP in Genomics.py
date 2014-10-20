"""
Implemented dynamic programming algorithms for determining both global and local alignments
of pairs of sequences.
"""

__author__ = 'liuyincheng'

DESKTOP = True

import math
import random
import urllib2

if DESKTOP:
    import matplotlib.pyplot as plt
#    import alg_project4_solution as student
else:
    import simpleplot
    import userXX_XXXXXXX as student


# URLs for data files
PAM50_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_PAM50.txt"
HUMAN_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_HumanEyelessProtein.txt"
FRUITFLY_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_FruitflyEyelessProtein.txt"
CONSENSUS_PAX_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_ConsensusPAXDomain.txt"
WORD_LIST_URL = "http://storage.googleapis.com/codeskulptor-assets/assets_scrabble_words3.txt"



###############################################
# provided code

def read_scoring_matrix(filename):
    """
    Read a scoring matrix from the file named filename.

    Argument:
    filename -- name of file containing a scoring matrix

    Returns:
    A dictionary of dictionaries mapping X and Y characters to scores
    """
    scoring_dict = {}
    scoring_file = urllib2.urlopen(filename)
    ykeys = scoring_file.readline()
    ykeychars = ykeys.split()
    for line in scoring_file.readlines():
        vals = line.split()
        xkey = vals.pop(0)
        scoring_dict[xkey] = {}
        for ykey, val in zip(ykeychars, vals):
            scoring_dict[xkey][ykey] = int(val)
    return scoring_dict


def read_protein(filename):
    """
    Read a protein sequence from the file named filename.

    Arguments:
    filename -- name of file containing a protein sequence

    Returns:
    A string representing the protein
    """
    protein_file = urllib2.urlopen(filename)
    protein_seq = protein_file.read()
    protein_seq = protein_seq.rstrip()
    return protein_seq


def read_words(filename):
    """
    Load word list from the file named filename.

    Returns a list of strings.
    """
    # load assets
    word_file = urllib2.urlopen(filename)

    # read in files as string
    words = word_file.read()

    # template lines and solution lines list of line string
    word_list = words.split('\n')
    print "Loaded a dictionary with", len(word_list), "words"
    return word_list


def build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score):
    """
    :param alphabet: a set of characters
    :param diag_score: the score for the diagonal entries besides dash
    :param off_diag_score: the score for off-diagonal entries besides dash
    :param dash_score: the score for any entry indexed by one or more dashes
    :return: a dictionary of dictionaries whose entries are indexed by pairs of characters in alphabet plus '-'.
    """
    indice = list(alphabet)
    indice.append("-")
    scoring_matrix = dict()
    for row_idx in indice:
        scoring_matrix[row_idx] = dict()
        for col_idx in indice:
            if row_idx == '-' or col_idx == '-':
                scoring_matrix[row_idx][col_idx] = dash_score
            elif row_idx == col_idx:
                scoring_matrix[row_idx][col_idx] = diag_score
            else:
                scoring_matrix[row_idx][col_idx] = off_diag_score
    return scoring_matrix


def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag):
    """
    :param seq_x and seq_y: two sequences of common alphabet with scoring matrix
    :param scoring_matrix:
    :param global_flag: if true then compute global pairwise alignment matrix.
    If false, compute local pairwise alignment matrix.
    :return: a list of lists representing alignment matrix
    """
    num_rows = len(seq_x)
    num_cols = len(seq_y)
    alignment_matrix = [ [0 for _i in range(num_cols + 1)] for _j in range(num_rows + 1)]  #Initial matrix with value zero
    for row in range(1, num_rows + 1):
        alignment_matrix[row][0] = scoring_matrix[seq_x[row - 1]]['-'] + alignment_matrix[row - 1][0]
        if global_flag == False:
            if alignment_matrix[row][0] < 0:
                alignment_matrix[row][0] = 0
    for col in range(1, num_cols + 1):
        alignment_matrix[0][col] = scoring_matrix['-'][seq_y[col - 1]] + alignment_matrix[0][col - 1]
        if global_flag == False:
            if alignment_matrix[0][col] < 0:
                alignment_matrix[0][col] = 0
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            alignment_matrix[row][col] = \
                max(alignment_matrix[row - 1][col -1 ] + scoring_matrix[seq_x[row - 1]][seq_y[col - 1]],
                    alignment_matrix[row - 1][col] + scoring_matrix[seq_x[row - 1]]['-'],
                    alignment_matrix[row][col - 1] + scoring_matrix['-'][seq_y[col - 1]])
            if global_flag == False:
                if alignment_matrix[row][col] < 0:
                    alignment_matrix[row][col] = 0
    return alignment_matrix


def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    computes a global alignment of seq_x and seq_y using the global alignment matrix
    :return a tuple of the form (score, align_x, align_y) where score is the score of the global alignment
    align_x and align_y
    """
    row = len(seq_x)
    col = len(seq_y)
    align_x = ''
    align_y = ''      # Initial empty alignment sequences
    while row > 0 and col > 0:
        if alignment_matrix[row][col] == (alignment_matrix[row - 1][col -1 ]
                                              + scoring_matrix[seq_x[row - 1]][seq_y[col - 1]]):
            align_x = seq_x[row - 1] + align_x
            align_y = seq_y[col - 1] + align_y
            row -= 1
            col-= 1
        elif alignment_matrix[row][col] == alignment_matrix[row - 1][col] + scoring_matrix[seq_x[row - 1]]['-']:
            align_x = seq_x[row - 1] + align_x
            align_y = '-' + align_y
            row -= 1
        else:
            align_x = '-' + align_x
            align_y = seq_y[col - 1] + align_y
            col -= 1
    while row > 0:
        align_x = seq_x[row - 1] + align_x
        align_y = '-' + align_y
        row -= 1
    while col > 0:
        align_x = '-' + align_x
        align_y = seq_y[col - 1] + align_y
        col -= 1
    score = sum(scoring_matrix[align_x[i]][align_y[i]] for i in range(len(align_x)))
    return (score, align_x, align_y)


def compute_local_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    compute a local alignment of seq_x and seq_y using the local alignment matrix alignment_matrix
    :return: a tuple of the form (score, align_x, align_y) where score is the score of the optimal
    local alignment align_x and align_y
    """
    score = 0
    init_row = len(seq_x)
    init_col = len(seq_y)
    for row in range(len(alignment_matrix)):
        for col in range(len(alignment_matrix[row])):
            if alignment_matrix[row][col] > score:
                score = alignment_matrix[row][col]
                init_row = row
                init_col = col
    row = init_row
    col = init_col
    align_x = ''
    align_y = ''      # Initial empty alignment sequences
    while row > 0 and col > 0 and alignment_matrix[row][col] > 0:
        if alignment_matrix[row][col] == (alignment_matrix[row - 1][col -1 ]
                                              + scoring_matrix[seq_x[row - 1]][seq_y[col - 1]]):
            align_x = seq_x[row - 1] + align_x
            align_y = seq_y[col - 1] + align_y
            row -= 1
            col-= 1
        elif alignment_matrix[row][col] == alignment_matrix[row - 1][col] + scoring_matrix[seq_x[row - 1]]['-']:
            align_x = seq_x[row - 1] + align_x
            align_y = '-' + align_y
            row -= 1
        else:
            align_x = '-' + align_x
            align_y = seq_y[col - 1] + align_y
            col -= 1
    return (score, align_x, align_y)


def generate_null_distribution(seq_x,seq_y, scoring_matrix, num_trials):
    """
    Return a dictionary scoring_distribution that represents an normalized distribution generated by
    performing the following process num_trials times
    """
    scoring_distribution = dict()
    for _i in range(num_trials):
        rand_y = list(seq_y)
        random.shuffle(rand_y)
        rand_y = ''.join(rand_y)
        score = compute_local_alignment(rand_y, seq_y, scoring_matrix,
                                        compute_alignment_matrix(rand_y, seq_y, scoring_matrix, False))[0]
        if score in scoring_distribution:
            scoring_distribution[score] += float(1) / float(num_trials)
        else:
            scoring_distribution[score] = float(1) / float(num_trials)
    return scoring_distribution


def check_spelling(checked_word, dist, word_list):
    alphabet = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    similarity_matrix = build_scoring_matrix(alphabet, 2, 1, 0)
    similar_words = []
    for word in word_list:
        alignment_matrix = compute_alignment_matrix(checked_word, word,similarity_matrix, True)
        score = compute_global_alignment(checked_word, word, similarity_matrix, alignment_matrix)[0]
        edit_distance = len(checked_word) + len(word) - score
        if edit_distance <= dist:
            similar_words.append(word)
    return similar_words


# Question 1&2
# human_protein = read_protein(HUMAN_EYELESS_URL)
# fly_protein = read_protein(FRUITFLY_EYELESS_URL)
# pax_domain = read_protein(CONSENSUS_PAX_URL)
# pam50_scoring_matrix = read_scoring_matrix(PAM50_URL)
# human_fly_alignment_matrix = compute_alignment_matrix(human_protein, fly_protein, pam50_scoring_matrix, False)
# human_fly_local_alignment = compute_local_alignment(human_protein, fly_protein, pam50_scoring_matrix,
#                                                     human_fly_alignment_matrix)
# human_local_alignment = human_fly_local_alignment[1].replace('-', '')
# fly_local_alignment = human_fly_local_alignment[2].replace('-', '')
# human_pam50_global_alignment = compute_global_alignment(human_local_alignment, pax_domain, pam50_scoring_matrix,
#                                                         compute_alignment_matrix(human_local_alignment, pax_domain,
#                                                                                  pam50_scoring_matrix, True))
# fly_pam50_global_alignment = compute_global_alignment(fly_local_alignment, pax_domain, pam50_scoring_matrix,
#                                                       compute_alignment_matrix(fly_local_alignment, pax_domain,
#                                                                                pam50_scoring_matrix, True))
# print human_pam50_global_alignment
# print fly_pam50_global_alignment

# Question 4:
# human_protein = read_protein(HUMAN_EYELESS_URL)
# fly_protein = read_protein(FRUITFLY_EYELESS_URL)
# pam50_scoring_matrix = read_scoring_matrix(PAM50_URL)
# normalized_dist = generate_null_distribution(human_protein, fly_protein, pam50_scoring_matrix, 1000)
# plt.bar(normalized_dist.keys(), normalized_dist.values())
# plt.xlabel("Score")
# plt.ylabel("Distribution")
# plt.title("Alignments Score Distribution \n between Human Protein and Shuffled Fruit Fly Protein")
# plt.show()

# Question 5:
# mean = 0.0
# expected_squared = 0.0
# for score in normalized_dist:
#     mean += score * normalized_dist[score]
#     expected_squared += (score ** 2) * normalized_dist[score]
# print "mean value =", mean
# standard_deviation = math.sqrt(expected_squared - mean ** 2)
# print "standard_deviation =", standard_deviation

# Question 6:
word_list = read_words(WORD_LIST_URL)
print check_spelling('humble', 1, word_list)
print check_spelling('firefly', 2, word_list)
