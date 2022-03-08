# AE-TPGG


This is a autoencoder based on a two-part-generalized-gamma distribution for improving single-cell RNA-seq data Analysis, implemented by python 3.

## Introduction
AE-TPGG uses a two-part-generalized-gamma model to capture the statistical characteristics of semi-continuous normalized data and adaptively explores the potential relationships between genes for promoting data imputation through the use of autoencoder.

## <a name="compilation"></a>  Installation

You can click [here](https://github.com/PUGEA/AE-TPGG) to download the AE-TPGG software. 


### Requirements:

*   AE-TPGG implementation uses [Scanpy](https://github.com/theislab/scanpy) to laod and pre-process the scRNA-seq data.
*   In AE-TPGG, the Python codes is implemented using [Keras](https://github.com/keras-team/keras) and its [TensorFlow](https://github.com/tensorflow/tensorflow) backend.

&nbsp;



### Example 1-Dimensionality Reduction

Here, we take the peripheral blood mononuclear cells (PMBC) 68K data as an example to show the specific process of dimensionality reduction using AE-TPGG. The 68k PBMC data is downloaded from
http://www.github.com/10XGenomics/single-cell-3prime-paper.


Step 1. Load data.

import scanpy.api as sc

adata = sc.read(path + 'matrix.mtx', cache=True).T  
adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]


Step 2. Feature selection. Select the top 1000 high variable genes by using the Scanpy package.
adata_temp = adata
sc.pp.normalize_per_cell(adata_temp)          # normalize with total UMI count per cell
filter_result = sc.pp.filter_genes_dispersion(adata_temp.X, flavor='cell_ranger', n_top_genes=1000, log=False)


Step 3. Data preprocessing. Convert the original count data into semi continuous form.
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata_filtered = adata[:, filter_result.gene_subset]


Step 4. Run AE-TPGG model and save the reduced data.

ae_tpgg(adata_ae_tpgg, threads=4)
np.savetxt('./pbmc68k_reduced_data', adata_filtered.obsm['X_ae_tpgg'])



Step 4. Display of visualization and clustering performance compared with tNSE.

![Image text](https://github.com/PUGEA/AE-TPGG/blob/main/Example_images/klein_tsne_2.png)
Figure 1. The t-SNE (left) and AE-TPGG (right) projections of 68K PBMCs, colored according to the 10 purified cell subtypes.


The tSNE visualizations of the Klein dataset. The fighures illustrates the results obtained from the Klein dataset, with the dropout outputs imputed by DAE-TPGM, respectively.

![Image text](https://github.com/PUGEA/AE-TPGG/blob/main/Example_images/klein_evaluation.png)

Figure 2. Clustering evaluation metrics including ACC, ARI, NMI, and F1 for the tSNE and AT-TPGG.





### Example 2-Imputation
Here, we take Klein dataset as an example to show the specific process of imputation using AE-TPGG. The download address of the Klein dataset is [here](https://scrnaseq-public-datasets.s3.amazonaws.com/scater-objects/klein.rds).



Step 1. Feature selection. Select the top 5000 high variable genes by using the Seurat package
source('klein_filterHVG.R')

Step 2. Data preprocessing. Convert the original count data into semi continuous form.

klein_data = sc.read_csv('./klein_filtered_data.csv')

sc.pp.normalize_per_cell(klein_data)

sc.pp.log1p(klein_data)

Step 3. Run AE-TPGG model and save the imputed data.

klein_data = ae_tpgg(klein_data, threads=4, copy=True, log1p=False, return_info=True)
tpgg_alpha = klein_data.X
tpgg_beta = klein_data.obsm['X_tpgg_beta']
tpgg_gamma = klein_data.obsm['X_tpgg_gamma']
tpgg_pi = klein_data.obsm['X_tpgg_pi']

from math import gamma

row_num, col_num = tpgg_alpha.shape
gg_mean = np.zeros((row_num, col_num))
tpgg_mean = np.zeros((row_num, col_num))

for i in range(row_num):
    for j in range(col_num):
        alpha_one = tpgg_alpha[i][j]
        beta_one = tpgg_beta[i][j]
        gamma_one = tpgg_gamma[i][j]
        pi_one = tpgg_pi[i][j]
        tpgg_mean[i][j] = (1-pi_one)*alpha_one * gamma(1 / gamma_one + beta_one) / gamma(beta_one)

klein_data.X = tpgg_mean
klein_data_df = pd.DataFrame(klein_data.X)
klein_data_df.to_csv('./klein_imputation_data.csv', sep=',', index=True, header=True)


Step 4. Display of visualization and clustering performance.

![Image text](https://github.com/PUGEA/AE-TPGG/blob/main/Example_images/klein_tsne_2.png)
Figure 2.The tSNE visualizations of the Klein dataset. The fighures illustrates the results obtained from the Klein dataset, with the dropout outputs imputed by AE-TPGG, respectively.


![Image text](https://github.com/PUGEA/AE-TPGG/blob/main/Example_images/klein_evaluation.png)

Figure 3. Clustering evaluation metrics including ACC, ARI, NMI, and F1 for the original data and imputed data of Klein dataset.


## Authors

The AE-TPGG algorithm is developed by Shuchang Zhao. 

## Contact information

For any query, please contact Shuchang Zhao via shuchangzhao@nuaa.edu.cn or Xuejun Liu via xuejun.liu@nuaa.edu.cn.
