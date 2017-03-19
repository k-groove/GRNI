from pandas import *
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

rowIDs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 1. Load the gene expression data (Gene_expression_1.csv) and the ground truth adjacency
# matrix (Adj_1.csv).
gene_exp_1 = pandas.read_csv('Gene_expression_1.csv')
ground_truth_adj = pandas.read_csv('Adj_1.csv', sep=',', header=None)

# 2. Compute pairwise correlation matrix, and show the matrix. see Fig. 1

gene_exp_1_corr = gene_exp_1.corr('pearson')
plt.show()
gene_exp_1_corr = gene_exp_1_corr.multiply(gene_exp_1_corr)
gene_exp_1_corr_matrix = gene_exp_1_corr.as_matrix()
# Get upper triangle matrix since anything below diagonal is a duplicate.
gene_exp_1_corr_matrix = np.triu(gene_exp_1_corr_matrix, 1)
print(tabulate(gene_exp_1_corr_matrix, showindex=rowIDs, headers=rowIDs))

# 3. Given the range of threshold (e.g., 0, 0.1, 0.2, 0.3, …, 0.9, 1), compare the adjacency
# matrices between the network and the ground truth.

ground_truth_adj = np.triu(ground_truth_adj, 1)  # Get upper triangle
print("Ground Truth")
print(tabulate(ground_truth_adj, showindex=rowIDs, headers=rowIDs))
for i in thresholds:
    temp1 = gene_exp_1_corr.clip_lower(i)  # Numbers < i are set to i
    gene_exp_1_corr_threshold = temp1.replace(i, 0)  # Number = i are replaced with 0
    gene_exp_1_corr_threshold_matrix = np.triu(gene_exp_1_corr_threshold.as_matrix(), 1)  # Get upper triangle

    print("Threshold: {}".format(i))
    print(tabulate(gene_exp_1_corr_threshold_matrix, showindex=rowIDs, headers=rowIDs))
    print("")
# 4. Compute a confusion matrix for each threshold

for i in thresholds:
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    temp2 = gene_exp_1_corr.clip_lower(i)  # Numbers < i are set to i
    gene_exp_1_corr_threshold = temp2.replace(i, 0)  # Number = i are replaced with 0
    gene_exp_1_corr_threshold_matrix = np.triu(gene_exp_1_corr_threshold.as_matrix(), 1)  # Get upper triangle
    for x in range(0, 10):
        for y in range(0, 10):
            if ground_truth_adj[x][y] > 0 and gene_exp_1_corr_threshold_matrix[x][y] > 0.0:
                true_pos += 1
            if ground_truth_adj[x][y] > 0 and gene_exp_1_corr_threshold_matrix[x][y] == 0.0:
                false_neg += 1
            if ground_truth_adj[x][y] == 0 and gene_exp_1_corr_threshold_matrix[x][y] == 0.0:
                true_neg += 1
            if ground_truth_adj[x][y] == 0 and gene_exp_1_corr_threshold_matrix[x][y] > 0.0:
                false_pos += 1
    print("Threshold: {}".format(i))
    print(tabulate([[true_pos, false_neg], [false_pos, true_neg]], showindex=['true', 'false'],
                   headers=['true', 'false']))
    print("")
# 5. Compute TPR and FPR for each threshold


# 6. Make a ROC plot. E.g., see Fig. 2
