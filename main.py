from pandas import *
from tabulate import tabulate

# 1. Load the gene expression data (Gene_expression_1.csv) and the ground truth adjacency
# matrix (Adj_1.csv).
gene_exp_1 = pandas.read_csv('Gene_expression_1.csv')
adj_1 = pandas.read_csv('Adj_1.csv')

# 2. Compute pairwise correlation matrix, and show the matrix. see Fig. 1
print(tabulate(adj_1))
rowIDs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
corr = gene_exp_1.corr('pearson')
corr_matrix = corr.as_matrix()
print(tabulate(corr_matrix, showindex=rowIDs, headers=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]))

# 3. Given the range of threshold (e.g., 0, 0.1, 0.2, 0.3, …, 0.9, 1), compare the adjacency
th_corr = corr.clip_lower(0.1)
print(tabulate(th_corr.as_matrix(), showindex=rowIDs, headers=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]))

# matrices between the network and the ground truth.


# 4. Compute a confusion matrix for each threshold

# 5. Compute TPR and FPR for each threshold

# 6. Make a ROC plot. E.g., see Fig. 2
