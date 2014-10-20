"""
Use Dynamic Programming in measuring the similarity between two sequences of characters.
"""
__author__ = 'liuyincheng'


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
