import numpy as np
import pandas as pd
import scipy as sp

m = pd.read_csv('./gene_sample_TPM.tsv', sep='\t', header=0, index_col=0)
h = pd.read_csv('./gene_celltype_TPM.tsv', sep='\t', header=0, index_col=0)
print(h.reindex(m.index))
ans = pd.read_csv('./sample_celltype_groundT.tsv', sep='\t', header=0, index_col=0)
m_tag = (h @ (ans / 100.0).T).reindex_like(m)
print(m_tag)
print("MSE:", ((m - m_tag) ** 2).to_numpy().mean())
print("Pearson Correlation Coefficient:", sp.stats.pearsonr(m, m_tag, axis=None).statistic)
print("Row sums mean:", (ans / 100.0).sum(axis=1).mean())
