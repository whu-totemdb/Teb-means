# Teb-means

## 1 Introduction

**Teb-means** is a **t**ime-efficient, f**e**derated, and **b**alanced $k$-means algorithm for high-dimensional data. This repo holds the source code and scripts for reproducing the key experiments of our paper: <u>Federated and Balanced Clustering for High-dimensional Data</u>.

## 2 Data Process

### 2.1 Preparation

- Create two empty folders `rawdata` and `output` under `./dataset/`.
- Put all the datasets into `./dataset/rawdata/`.
- Then you can get a directory overview as follows:

```
dataset
├── dataprocess
│   ├── data_process.py
│   ├── dataset_configs.ini
│   └── example_config.ini
├── output
└── rawdata
```

The datasets we use are all high-dimensional, with brief information as shown in the list below:

|                           Datasets                           | Dataset Scale | Dimension |                         Description                          |
| :----------------------------------------------------------: | :-----------: | :-------: | :----------------------------------------------------------: |
| **[Credit](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** |     0.28M     |      30     | Credit consists of 284,807 anonymized transactions, some of which are labeled as fraudulent, and contains 30 features. Each transaction includes both transaction-specific attributes and user-related characteristics. The dataset can be modeled such that each transaction's features (particularly those related to the initiating user) are distributed across different parties, such as banks and merchants. Hence, we can construct a federated and balanced $k$-means to support ANN. This will enable the identification of similar historical transactions for fraud detection when a new transaction arrives. |
| **[MTG](https://ffiec.cfpb.gov/data-browser/)** |     5.98M     |     53    | MTG provides detailed annual information on mortgage loan applications and approvals from U.S. financial institutions, including applicant backgrounds, loan types, and approval statuses. For individuals without prior loans, we can identify similar borrowers to assess their credit risk. Since borrowers’ information is distributed across different financial institutions, a single institution may lack sufficient data to accurately assess an individual’s credit risk. To address this, applying balanced and federated $k$-means across institutions to build an index and identify similar client offers an effective way to estimate loan risk. |
|      **[Census](https://www.argoverse.org/av2.html)**      |    5M     |     69      | This dataset consists of 69 features collected from the U.S. Census. Many companies can leverage this data for market segmentation, policy development, and social planning. However, since much of the data is private, companies cannot directly extract information to support decision-making, such as financial product recommendations. In this context, our method provides a suitable solution by offering an ANN search to identify individuals similar to the ones being considered. |
| **[Game](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox)** |    3M       |    79       | The data in Game originates from three platforms that are unable to share data directly due to privacy or regulatory constraints. Each platform contains information such as user attributes, game features, and user-generated content like reviews and ratings. When a new game is launched on a platform, pricing or promotion decisions can be guided by identifying similar games through ANN search using early-access information. To support this, it is necessary to construct a clustering-based index using federated and balanced $k$-means, which enables effective ANN search while adhering to data-sharing constraints. |
| **[Job](https://data.cityofnewyork.us/Housing-Development/DOB-Job-Application-Filings/ic3t-wcy2/about_data)** |    2.7M     |     128        | The dataset used in this study is sourced from NYC Open Data and consists of job application records submitted to the Department of Buildings. Each entry represents an individual applicant and contains 96 attributes, including some textual fields. After preprocessing and converting all features into numerical format, the dataset includes a total of 128 features. This dataset allows us to model the scenario where applicant features are distributed across multiple companies. When a new applicant arrives, our algorithm leverages the constructed index to perform ANN search, enabling the assessment of their similarity to existing profiles and determining their suitability for the job. |
|            **[Retail](https://www.kaggle.com/datasets/ricgomes/global-fashion-retail-stores-dataset/data?select=customers.csv)**             |      2.7M     |     597   | The dataset provides information on customers from global fashion retail stores, including details such as customer ID, characteristics of purchased products (e.g., product price, discount, and store location), and demographic attributes. These features, however, belong to different stores and are considered private data, thus not to be shared externally. Therefore, we can leverage this dataset to construct an index using our proposed method, enabling support for ANN, as discussed in our introduction. |
|            **[Amazon](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv)**             |      1.24M      |     2400       | The dataset comprises customer records from international fashion retail outlets, encompassing attributes such as customer identifiers, characteristics of purchased items (including pricing, applied discounts, and store locations), as well as demographic information. Due to the proprietary nature of these features, which are specific to individual stores, the data is considered confidential and not suitable for external distribution. Consequently, we utilize this dataset to develop an index through our proposed approach, thereby enabling approximate nearest neighbor (ANN) search capabilities as outlined in the introduction. |
|            **[MovieLens](https://grouplens.org/datasets/movielens/)**             |     21M      |     5000      | MovieLens is a widely used dataset that contains user ratings for movies. We can restructure the data so that each row represents a user and each column corresponds to a different movie. Each entry in the matrix then reflects a user's rating for a specific movie. Based on this, we consider a scenario where several companies share the same group of users, but each company holds the copyright for different sets of movies. These users watch and rate movies across the different platforms owned by these companies. When a new user arrives, we can recommend movies they have not yet seen. At this point, we can use Teb-means to construct a clustering-based index and employ ANN search to find similar users, recommending movies they have rated highly. |

### 2.2 Data Process

- All the code for data processing is in `./dataset/dataprocess/data_process.py`.
- Set the basic information (e.g., path, separator) in `./dataset/dataprocess/dataset_configs.ini`.
- Set the `max_points` and `num_clusters` for each dataset in `./dataset/dataprocess/example_config.ini`.
- Update the `dataset_list` in the main function of `data_process.py` based on the alias you set in `dataset_configs.ini`.
- Run the main function to start data processing, and you can get the output in `./dataset/output/`.

## 3 Comparison Algorithms

1. [Least squares quantization in PCM](https://hal.science/hal-04614938/document)
1. [Balanced clustering with least square regression](https://ojs.aaai.org/index.php/AAAI/article/view/10877)
1. [Fast clustering with flexible balance constraints](https://ieeexplore.ieee.org/abstract/document/8621917/)
1. [Coordinate Descent Method for k-means](https://ieeexplore.ieee.org/abstract/document/9444882/)
1. [F3KM: Federated, Fair, and Fast k-means](https://dl.acm.org/doi/abs/10.1145/3626728)

## 4 How to Run Teb-means

You can run `Teb-means` and all the comparison algorithms in `./run.m`.

**Parameter configuration:**

```matlab
% 0.test time
num_runs = 3;

% 1.comparison algorithms
methods = {'Lloyd', 'CDKM', 'FCFC', 'BCLS_ALM', 'F3KM', 'Teb' };

% 2.datasets
datasets = {'rename this term into your datasets'};

% 3.thread number
threads_set = 1;

% 4.the k value size
clusters_size = {3,4,5,6,7,8,9,10};

% 5.different rho/eta value
eta_set = [0.83];
 
% 6.block size
blocks_size = [1,2,4,8,16,32,64,128,256,512];
```

