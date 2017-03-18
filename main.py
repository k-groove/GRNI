from pandas import *
from tabulate import tabulate
import numpy as np

rowIDs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# 1. Load the gene expression data (Gene_expression_1.csv) and the ground truth adjacency
# matrix (Adj_1.csv).
gene_exp_1 = pandas.read_csv('Gene_expression_1.csv')
adj_1 = pandas.read_csv('Adj_1.csv', sep=',', header=None)

# 2. Compute pairwise correlation matrix, and show the matrix. see Fig. 1
print(tabulate(adj_1))
gene_exp_1_corr = gene_exp_1.corr('pearson')
gene_exp_1_corr_matrix = gene_exp_1_corr.as_matrix()

print(tabulate(gene_exp_1_corr_matrix, showindex=rowIDs, headers=rowIDs))
print(np.triu(np.ones(gene_exp_1_corr.shape)).astype(np.bool))
gene_exp_1_corr = gene_exp_1_corr.where(np.triu(np.ones(gene_exp_1_corr.shape)).astype(np.bool))
print(tabulate(gene_exp_1_corr, showindex=rowIDs, headers=rowIDs))

# 3. Given the range of threshold (e.g., 0, 0.1, 0.2, 0.3, …, 0.9, 1), compare the adjacency
# matrices between the network and the ground truth.
# threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# for i in threshold:
#     gene_exp_1_corr = gene_exp_1_corr.clip_lower(i)
#     print("Threshold: {}".format(i))
#     gene_exp_1_corr.replace(i, 0)
#     print(tabulate(gene_exp_1_corr.as_matrix(), showindex=rowIDs, headers=rowIDs))

# 4. Compute a confusion matrix for each threshold
# for i in threshold:
#     y_actu = Series(corr.clip_lower(j), name='Actual')
#     y_pred = Series(adj_1, name='Predicted')
#     corr_confusion = crosstab(y_actu, y_pred)
#     print(corr_confusion)

# 5. Compute TPR and FPR for each threshold


# 6. Make a ROC plot. E.g., see Fig. 2
