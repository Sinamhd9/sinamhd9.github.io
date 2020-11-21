# Problem Description

It is important to ask the right question before trying to solve it! Can we predict mechanism of action (MoA) of a drug based on gene expression and cell viability data? Or better to ask first, what is mechansim of action? The term mechanism of action means the biochemical interactions through which a drug generates its pharmacological effect. Scientists know many MoAs of drugs, for example, an antidepressant may have a selective serotonin reuptake inhibitor (SSRI), which affects the brain serotonin level. In this project we are going to train a model that classifies drugs based on their biological activity. The dataset consists of different features of gene expression data, and cell viability data as well as multiple targets of mechansim of action (MoA). This problem is a multilabel classification, which means we have multiple targets (not multiple classes). In this project, we will first perform explanatory data analysis and then train a model using deep neural networks with Keras. We will do a bit model evaluation at the end.


```python
# Importing useful libraries
import warnings
warnings.filterwarnings("ignore")

# Adding iterative-stratification 
# Select add data from the right menu and search for iterative-stratification, then add it to your kernel.
import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from time import time
import datetime
import gc

import numpy as np
import pandas as pd 

# ML tools 
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf 
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from sklearn.metrics import log_loss
from tensorflow_addons.layers import WeightNormalization
# Setting random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
sns.set(font_scale=1.2)


```

# Exploratory Data Analysis (EDA)


## Example data
First, we are going to see the train and test data size and some of their examples. Please note there are two different target dataframes, non-scored and scored. The non-scored ones are not used for scoring, but we can make use of them to pretrain our network
<a href="https://www.kaggle.com/kailex/moa-transfer-recipe-with-smoothing"> [1]</a>
 


```python
df_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
display(df_train.head(3))
print('train data size', df_train.shape)

df_target_ns = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
display(df_target_ns.head(3))
print('train target nonscored size', df_target_ns.shape)


df_target_s = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
display(df_target_s.head(3))
print('train target scored size', df_target_s.shape)


df_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
display(df_test.head(3))
print('test data size', df_test.shape)

df_sample = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
display(df_sample.head(3))
print('sample submission size', df_sample.shape)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>c-90</th>
      <th>c-91</th>
      <th>c-92</th>
      <th>c-93</th>
      <th>c-94</th>
      <th>c-95</th>
      <th>c-96</th>
      <th>c-97</th>
      <th>c-98</th>
      <th>c-99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>24</td>
      <td>D1</td>
      <td>1.0620</td>
      <td>0.5577</td>
      <td>-0.2479</td>
      <td>-0.6208</td>
      <td>-0.1944</td>
      <td>-1.0120</td>
      <td>...</td>
      <td>0.2862</td>
      <td>0.2584</td>
      <td>0.8076</td>
      <td>0.5523</td>
      <td>-0.1912</td>
      <td>0.6584</td>
      <td>-0.3981</td>
      <td>0.2139</td>
      <td>0.3801</td>
      <td>0.4176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>72</td>
      <td>D1</td>
      <td>0.0743</td>
      <td>0.4087</td>
      <td>0.2991</td>
      <td>0.0604</td>
      <td>1.0190</td>
      <td>0.5207</td>
      <td>...</td>
      <td>-0.4265</td>
      <td>0.7543</td>
      <td>0.4708</td>
      <td>0.0230</td>
      <td>0.2957</td>
      <td>0.4899</td>
      <td>0.1522</td>
      <td>0.1241</td>
      <td>0.6077</td>
      <td>0.7371</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>trt_cp</td>
      <td>48</td>
      <td>D1</td>
      <td>0.6280</td>
      <td>0.5817</td>
      <td>1.5540</td>
      <td>-0.0764</td>
      <td>-0.0323</td>
      <td>1.2390</td>
      <td>...</td>
      <td>-0.7250</td>
      <td>-0.6297</td>
      <td>0.6103</td>
      <td>0.0223</td>
      <td>-1.3240</td>
      <td>-0.3174</td>
      <td>-0.6417</td>
      <td>-0.2187</td>
      <td>-1.4080</td>
      <td>0.6931</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 876 columns</p>
</div>


    train data size (23814, 876)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>abc_transporter_expression_enhancer</th>
      <th>abl_inhibitor</th>
      <th>ace_inhibitor</th>
      <th>acetylcholine_release_enhancer</th>
      <th>adenosine_deaminase_inhibitor</th>
      <th>adenosine_kinase_inhibitor</th>
      <th>adenylyl_cyclase_inhibitor</th>
      <th>age_inhibitor</th>
      <th>alcohol_dehydrogenase_inhibitor</th>
      <th>...</th>
      <th>ve-cadherin_antagonist</th>
      <th>vesicular_monoamine_transporter_inhibitor</th>
      <th>vitamin_k_antagonist</th>
      <th>voltage-gated_calcium_channel_ligand</th>
      <th>voltage-gated_potassium_channel_activator</th>
      <th>voltage-gated_sodium_channel_blocker</th>
      <th>wdr5_mll_interaction_inhibitor</th>
      <th>wnt_agonist</th>
      <th>xanthine_oxidase_inhibitor</th>
      <th>xiap_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 403 columns</p>
</div>


    train target nonscored size (23814, 403)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor</th>
      <th>11-beta-hsd1_inhibitor</th>
      <th>acat_inhibitor</th>
      <th>acetylcholine_receptor_agonist</th>
      <th>acetylcholine_receptor_antagonist</th>
      <th>acetylcholinesterase_inhibitor</th>
      <th>adenosine_receptor_agonist</th>
      <th>adenosine_receptor_antagonist</th>
      <th>adenylyl_cyclase_activator</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 207 columns</p>
</div>


    train target scored size (23814, 207)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>c-90</th>
      <th>c-91</th>
      <th>c-92</th>
      <th>c-93</th>
      <th>c-94</th>
      <th>c-95</th>
      <th>c-96</th>
      <th>c-97</th>
      <th>c-98</th>
      <th>c-99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>trt_cp</td>
      <td>24</td>
      <td>D1</td>
      <td>-0.5458</td>
      <td>0.1306</td>
      <td>-0.5135</td>
      <td>0.4408</td>
      <td>1.5500</td>
      <td>-0.1644</td>
      <td>...</td>
      <td>0.0981</td>
      <td>0.7978</td>
      <td>-0.143</td>
      <td>-0.2067</td>
      <td>-0.2303</td>
      <td>-0.1193</td>
      <td>0.0210</td>
      <td>-0.0502</td>
      <td>0.1510</td>
      <td>-0.7750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>trt_cp</td>
      <td>72</td>
      <td>D1</td>
      <td>-0.1829</td>
      <td>0.2320</td>
      <td>1.2080</td>
      <td>-0.4522</td>
      <td>-0.3652</td>
      <td>-0.3319</td>
      <td>...</td>
      <td>-0.1190</td>
      <td>-0.1852</td>
      <td>-1.031</td>
      <td>-1.3670</td>
      <td>-0.3690</td>
      <td>-0.5382</td>
      <td>0.0359</td>
      <td>-0.4764</td>
      <td>-1.3810</td>
      <td>-0.7300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_002429b5b</td>
      <td>ctl_vehicle</td>
      <td>24</td>
      <td>D1</td>
      <td>0.1852</td>
      <td>-0.1404</td>
      <td>-0.3911</td>
      <td>0.1310</td>
      <td>-1.4380</td>
      <td>0.2455</td>
      <td>...</td>
      <td>-0.2261</td>
      <td>0.3370</td>
      <td>-1.384</td>
      <td>0.8604</td>
      <td>-1.9530</td>
      <td>-1.0140</td>
      <td>0.8662</td>
      <td>1.0160</td>
      <td>0.4924</td>
      <td>-0.1942</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 876 columns</p>
</div>


    test data size (3982, 876)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor</th>
      <th>11-beta-hsd1_inhibitor</th>
      <th>acat_inhibitor</th>
      <th>acetylcholine_receptor_agonist</th>
      <th>acetylcholine_receptor_antagonist</th>
      <th>acetylcholinesterase_inhibitor</th>
      <th>adenosine_receptor_agonist</th>
      <th>adenosine_receptor_antagonist</th>
      <th>adenylyl_cyclase_activator</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_002429b5b</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 207 columns</p>
</div>


    sample submission size (3982, 207)
    

## Missing values
Let's see if there are any missing values, and see some information about our data types. 


```python
print(df_train.isnull().sum().any()) # True if there are missing values
print(df_train.info())
```

    False
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23814 entries, 0 to 23813
    Columns: 876 entries, sig_id to c-99
    dtypes: float64(872), int64(1), object(3)
    memory usage: 159.2+ MB
    None
    

There are no missing values; there are 872 float dtypes, 1 integer and 3 objects. Let's print the latter ones.


```python
display(df_train.select_dtypes('int64').head(3))
display(df_train.select_dtypes('object').head(3))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cp_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_dose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>trt_cp</td>
      <td>D1</td>
    </tr>
  </tbody>
</table>
</div>


## Features
Let's visualize some features randomly:
### Gene expression features


```python
g_features = [cols for cols in df_train.columns if cols.startswith('g-')]
```


```python
color = ['dimgray','navy','purple','orangered', 'red', 'green' ,'mediumorchid', 'khaki', 'salmon', 'blue','cornflowerblue','mediumseagreen']
 
color_ind=0
n_row = 6
n_col = 3
n_sub = 1 
plt.rcParams["legend.loc"] = 'upper right'
fig = plt.figure(figsize=(8,14))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
for i in (np.arange(0,6,1)):
    plt.subplot(n_row, n_col, n_sub)
    sns.kdeplot(df_train.loc[:,g_features[i]],color=color[color_ind],shade=True,
                 label=['mean:'+str('{:.2f}'.format(df_train.loc[:,g_features[i]].mean()))
                        +'  ''std: '+str('{:.2f}'.format(df_train.loc[:,g_features[i]].std()))])
    
    plt.xlabel(g_features[i])
    plt.legend()                    
    n_sub+=1
    color_ind+=1
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_11_0.png)


### Cell viability features


```python
c_features = [cols for cols in df_train.columns if cols.startswith('c-')]
```


```python
n_row = 6
n_col = 3
n_sub = 1 
fig = plt.figure(figsize=(8,14))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
plt.rcParams["legend.loc"] = 'upper left'
for i in (np.arange(0,6,1)):
    plt.subplot(n_row, n_col, n_sub)
    sns.kdeplot(df_train.loc[:,c_features[i]],color=color[color_ind],shade=True,
                 label=['mean:'+str('{:.2f}'.format(df_train.loc[:,c_features[i]].mean()))
                        +'  ''std: '+str('{:.2f}'.format(df_train.loc[:,c_features[i]].std()))])
    
    plt.xlabel(c_features[i])
    plt.legend()                    
    n_sub+=1
    color_ind+=1
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_14_0.png)


It seems data are somehow normalized and also clipped at -10, 10. Please see this great discussion here: <a href="https://www.kaggle.com/c/lish-moa/discussion/184005"> [2] </a>

### Cp_time and cp_dose

cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low which are D1 and D2).


```python
fig = plt.figure(figsize=(10,4))
plt.subplots_adjust(right=1.3)
plt.subplot(1, 2, 1)
sns.countplot(df_train['cp_time'],palette='nipy_spectral')
plt.subplot(1, 2, 2)
sns.countplot(df_train['cp_dose'],palette='nipy_spectral')
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_17_0.png)


We can see there are almost the same number of examples in each treatment duration and dosage features.


## Exploring some relationships

Next, we can use stripplot to show the relationship of a feature and a target with respect to dosage and time. Since, this is a multilabel probelm, we only show one label here, which is target 71. We will see later this target is contributing the most to the loss. For the feature, we chose two random g and c features. You may wanna do this with other features and labels to get more insight. 


```python
train_copy= df_train.copy()
train_copy['target_71'] = df_target_s.iloc[:,72] # sig_id is included
fig = plt.figure(figsize=(16,8))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy , x='cp_time', y= 'g-3',color='red', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'g-3',color='red', hue='target_71',ax=ax2)
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_21_0.png)



```python
fig = plt.figure(figsize=(16,8))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy, x='cp_time', y= 'c-1',color='yellow', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'c-1',color='yellow', hue='target_71',ax=ax2)
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_22_0.png)


Or we can do the same process with the mean of g and c features. For example, here we plot the mean of g and c features with respect to a target, dosage and time. 


```python
train_copy['g_mean'] = train_copy.loc[:, g_features].mean(axis=1) 
fig = plt.figure(figsize=(16,10))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy , x='cp_time', y= 'g_mean',color='red', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'g_mean', color='red', hue='target_71',ax=ax2)
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_24_0.png)



```python
train_copy['c_mean'] = train_copy.loc[:, c_features].mean(axis=1) 
fig = plt.figure(figsize=(16,10))
plt.subplots_adjust(right=1.1,top=1.1)
ax1 = fig.add_subplot(121)
sns.stripplot(data= train_copy, x='cp_time', y= 'c_mean',color='yellow', hue='target_71',ax=ax1)
ax2 = fig.add_subplot(122)
sns.stripplot(data= train_copy , x='cp_dose', y= 'c_mean', color='yellow', hue='target_71',ax=ax2)
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_25_0.png)


We can get some insights from the figures above and apply it in our [Preprocessing](#26) step 

## Targets
Below are some scored targets which are used to train the main model. As we can see, the targets are very imbalanced and there are only a few positive examples in some labels. 


```python
target_s_copy = df_target_s.copy()
target_s_copy.drop('sig_id', axis=1, inplace=True)
n_row = 20
n_col = 4 
n_sub = 1   
fig = plt.figure(figsize=(20,50))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
for i in np.random.choice(np.arange(0,target_s_copy.shape[1],1),n_row):
    plt.subplot(n_row, n_col, n_sub)
    sns.countplot(y=target_s_copy.iloc[:, i],palette='nipy_spectral',orient='h')
    
    plt.legend()                    
    n_sub+=1
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_28_0.png)


Let's see the 20 largest positive number of labels in the scored targets. 


```python
plt.figure(figsize=(10,10))
target_s_copy.sum().sort_values()[-20:].plot(kind='barh',color='mediumseagreen')
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_30_0.png)


And here are some non-scored targets. We can see that some labels do no have positive examples at all.


```python
target_ns_copy = df_target_ns.copy()
target_ns_copy.drop('sig_id', axis=1, inplace=True)
n_row = 20
n_col = 4 
n_sub = 1   
fig = plt.figure(figsize=(20,50))
plt.subplots_adjust(left=-0.3, right=1.3,bottom=-0.3,top=1.3)
for i in np.random.choice(np.arange(0,target_ns_copy.shape[1],1),n_row):
    plt.subplot(n_row, n_col, n_sub)
    sns.countplot(y=target_ns_copy.iloc[:, i],palette='magma',orient='h')
    
    plt.legend()                    
    n_sub+=1
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_32_0.png)


And here is the 20 largest positive number of labels in the non-scored targets. 


```python
plt.figure(figsize=(10,10))
target_ns_copy.sum().sort_values()[-20:].plot(kind='barh',color='purple')
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_34_0.png)


As we can see, there are fewer positive examples in non-scored dataset.

## Preprocessing and feature engineering

The control group is defined as the group in an experiment or study that does not have the desired effect or MoAs here; which means the target labels are zero for them. I will drop the data for this group, and we will later set all predictions of this group to zero.

We will keep track of the control group (ctl_vehicle) indexes. 
I dropped cp_type column and mapped the values of time and dose features. I performed some feature engineering based on the insights I got from the "Exploring some relationships" part. <br>
Update : I added the methods of Rankgauss scaler and PCA from this great kernel : <a href='https://www.kaggle.com/kushal1506/moa-pytorch-0-01859-rankgauss-pca-nn?scriptVersionId=44558776' >[3] 


```python
ind_tr = df_train[df_train['cp_type']=='ctl_vehicle'].index
ind_te = df_test[df_test['cp_type']=='ctl_vehicle'].index
```


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
transformer = QuantileTransformer(n_quantiles=100,random_state=42, output_distribution="normal")

def preprocess(df):
    df['cp_time'] = df['cp_time'].map({24:1, 48:2, 72:3})
    df['cp_dose'] = df['cp_dose'].map({'D1':0, 'D2':1})
    g_features = [cols for cols in df.columns if cols.startswith('g-')]
    c_features = [cols for cols in df.columns if cols.startswith('c-')]
    for col in (g_features + c_features):
        vec_len = len(df[col].values)
        raw_vec = df[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        df[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    return df

X = preprocess(df_train)
X_test = preprocess(df_test)

display(X.head(5))
print('Train data size', X.shape)
display(X_test.head(3))
print('Test data size', X_test.shape)
y = df_target_s.drop('sig_id', axis=1)
display(y.head(3))
print('target size', y.shape)
y0 =  df_target_ns.drop('sig_id', axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>c-90</th>
      <th>c-91</th>
      <th>c-92</th>
      <th>c-93</th>
      <th>c-94</th>
      <th>c-95</th>
      <th>c-96</th>
      <th>c-97</th>
      <th>c-98</th>
      <th>c-99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>1.134849</td>
      <td>0.907687</td>
      <td>-0.416385</td>
      <td>-0.966814</td>
      <td>-0.254723</td>
      <td>-1.017473</td>
      <td>...</td>
      <td>0.410974</td>
      <td>0.364819</td>
      <td>1.291804</td>
      <td>0.835350</td>
      <td>-0.240101</td>
      <td>1.021706</td>
      <td>-0.499652</td>
      <td>0.317989</td>
      <td>0.545662</td>
      <td>0.641339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>0.119282</td>
      <td>0.681738</td>
      <td>0.272399</td>
      <td>0.080113</td>
      <td>1.205169</td>
      <td>0.686517</td>
      <td>...</td>
      <td>-0.520372</td>
      <td>1.127405</td>
      <td>0.716111</td>
      <td>0.054620</td>
      <td>0.412012</td>
      <td>0.744215</td>
      <td>0.210242</td>
      <td>0.179684</td>
      <td>0.919161</td>
      <td>1.165833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>trt_cp</td>
      <td>2</td>
      <td>0</td>
      <td>0.779973</td>
      <td>0.946463</td>
      <td>1.425350</td>
      <td>-0.132928</td>
      <td>-0.006122</td>
      <td>1.492493</td>
      <td>...</td>
      <td>-0.828896</td>
      <td>-0.740965</td>
      <td>0.953239</td>
      <td>0.053633</td>
      <td>-1.213056</td>
      <td>-0.394118</td>
      <td>-0.758652</td>
      <td>-0.277635</td>
      <td>-1.123088</td>
      <td>1.089235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>trt_cp</td>
      <td>2</td>
      <td>0</td>
      <td>-0.734910</td>
      <td>-0.274641</td>
      <td>-0.438509</td>
      <td>0.759097</td>
      <td>2.346330</td>
      <td>-0.858153</td>
      <td>...</td>
      <td>-1.419080</td>
      <td>-0.756098</td>
      <td>-1.652159</td>
      <td>-1.250427</td>
      <td>-0.947092</td>
      <td>-1.231225</td>
      <td>-1.325697</td>
      <td>-0.977581</td>
      <td>-0.485139</td>
      <td>-0.915321</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>1</td>
      <td>-0.452718</td>
      <td>-0.477513</td>
      <td>0.972316</td>
      <td>0.970731</td>
      <td>1.463427</td>
      <td>-0.869555</td>
      <td>...</td>
      <td>0.018697</td>
      <td>0.002153</td>
      <td>1.051051</td>
      <td>1.682158</td>
      <td>0.796356</td>
      <td>-0.378324</td>
      <td>0.153519</td>
      <td>0.428792</td>
      <td>-0.475464</td>
      <td>1.119408</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 876 columns</p>
</div>


    Train data size (23814, 876)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>c-90</th>
      <th>c-91</th>
      <th>c-92</th>
      <th>c-93</th>
      <th>c-94</th>
      <th>c-95</th>
      <th>c-96</th>
      <th>c-97</th>
      <th>c-98</th>
      <th>c-99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>-0.755294</td>
      <td>0.214796</td>
      <td>-0.774511</td>
      <td>0.705349</td>
      <td>1.564580</td>
      <td>-0.194968</td>
      <td>...</td>
      <td>0.116890</td>
      <td>1.194732</td>
      <td>-0.195260</td>
      <td>-0.298039</td>
      <td>-0.301677</td>
      <td>-0.128405</td>
      <td>-0.036089</td>
      <td>-0.094841</td>
      <td>0.153376</td>
      <td>-0.920303</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>-0.186773</td>
      <td>0.374643</td>
      <td>1.176402</td>
      <td>-0.652299</td>
      <td>-0.546638</td>
      <td>-0.403447</td>
      <td>...</td>
      <td>-0.183415</td>
      <td>-0.269845</td>
      <td>-1.059129</td>
      <td>-1.317859</td>
      <td>-0.466468</td>
      <td>-0.653500</td>
      <td>-0.015480</td>
      <td>-0.621168</td>
      <td>-1.191952</td>
      <td>-0.880456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_002429b5b</td>
      <td>ctl_vehicle</td>
      <td>1</td>
      <td>0</td>
      <td>0.351229</td>
      <td>-0.155308</td>
      <td>-0.613250</td>
      <td>0.255053</td>
      <td>-1.762152</td>
      <td>0.333321</td>
      <td>...</td>
      <td>-0.328362</td>
      <td>0.463214</td>
      <td>-1.233935</td>
      <td>1.310102</td>
      <td>-1.429491</td>
      <td>-1.093671</td>
      <td>1.239167</td>
      <td>1.577578</td>
      <td>0.681627</td>
      <td>-0.241098</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 876 columns</p>
</div>


    Test data size (3982, 876)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>5-alpha_reductase_inhibitor</th>
      <th>11-beta-hsd1_inhibitor</th>
      <th>acat_inhibitor</th>
      <th>acetylcholine_receptor_agonist</th>
      <th>acetylcholine_receptor_antagonist</th>
      <th>acetylcholinesterase_inhibitor</th>
      <th>adenosine_receptor_agonist</th>
      <th>adenosine_receptor_antagonist</th>
      <th>adenylyl_cyclase_activator</th>
      <th>adrenergic_receptor_agonist</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 206 columns</p>
</div>


    target size (23814, 206)
    


```python
# Please see reference 3 for this part
g_features = [cols for cols in X.columns if cols.startswith('g-')]
n_comp = 0.95

data = pd.concat([pd.DataFrame(X[g_features]), pd.DataFrame(X_test[g_features])])
data2 = (PCA(0.95, random_state=42).fit_transform(data[g_features]))
train2 = data2[:X.shape[0]]
test2 = data2[-X_test.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_g-{i}' for i in range(data2.shape[1])])
test2 = pd.DataFrame(test2, columns=[f'pca_g-{i}' for i in range(data2.shape[1])])

X = pd.concat((X, train2), axis=1)
X_test = pd.concat((X_test, test2), axis=1)

c_features = [cols for cols in X.columns if cols.startswith('c-')]
n_comp = 0.95

data = pd.concat([pd.DataFrame(X[c_features]), pd.DataFrame(X_test[c_features])])
data2 = (PCA(0.95, random_state=42).fit_transform(data[c_features]))
train2 = data2[:X.shape[0]]
test2 = data2[-X_test.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_c-{i}' for i in range(data2.shape[1])])
test2 = pd.DataFrame(test2, columns=[f'pca_c-{i}' for i in range(data2.shape[1])])

X = pd.concat((X, train2), axis=1)
X_test = pd.concat((X_test, test2), axis=1)

display(X.head(2))
display(X_test.head(2))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>pca_c-74</th>
      <th>pca_c-75</th>
      <th>pca_c-76</th>
      <th>pca_c-77</th>
      <th>pca_c-78</th>
      <th>pca_c-79</th>
      <th>pca_c-80</th>
      <th>pca_c-81</th>
      <th>pca_c-82</th>
      <th>pca_c-83</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>1.134849</td>
      <td>0.907687</td>
      <td>-0.416385</td>
      <td>-0.966814</td>
      <td>-0.254723</td>
      <td>-1.017473</td>
      <td>...</td>
      <td>-0.429101</td>
      <td>0.550838</td>
      <td>-0.892721</td>
      <td>0.135728</td>
      <td>1.329258</td>
      <td>0.522436</td>
      <td>0.075909</td>
      <td>0.300444</td>
      <td>-0.958183</td>
      <td>-0.222714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>0.119282</td>
      <td>0.681738</td>
      <td>0.272399</td>
      <td>0.080113</td>
      <td>1.205169</td>
      <td>0.686517</td>
      <td>...</td>
      <td>-0.683204</td>
      <td>-0.247614</td>
      <td>0.582455</td>
      <td>-0.287878</td>
      <td>-0.289009</td>
      <td>0.235928</td>
      <td>-1.066457</td>
      <td>-0.229447</td>
      <td>-0.269402</td>
      <td>-0.404329</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1564 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>pca_c-74</th>
      <th>pca_c-75</th>
      <th>pca_c-76</th>
      <th>pca_c-77</th>
      <th>pca_c-78</th>
      <th>pca_c-79</th>
      <th>pca_c-80</th>
      <th>pca_c-81</th>
      <th>pca_c-82</th>
      <th>pca_c-83</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>-0.755294</td>
      <td>0.214796</td>
      <td>-0.774511</td>
      <td>0.705349</td>
      <td>1.564580</td>
      <td>-0.194968</td>
      <td>...</td>
      <td>0.886659</td>
      <td>0.060739</td>
      <td>0.756908</td>
      <td>1.240647</td>
      <td>0.011547</td>
      <td>-0.183502</td>
      <td>1.374422</td>
      <td>0.314476</td>
      <td>-0.640933</td>
      <td>-0.819822</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>-0.186773</td>
      <td>0.374643</td>
      <td>1.176402</td>
      <td>-0.652299</td>
      <td>-0.546638</td>
      <td>-0.403447</td>
      <td>...</td>
      <td>0.408430</td>
      <td>-0.817178</td>
      <td>-0.219832</td>
      <td>0.355091</td>
      <td>-1.072866</td>
      <td>-0.336214</td>
      <td>0.421539</td>
      <td>0.762212</td>
      <td>-0.324633</td>
      <td>1.232997</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1564 columns</p>
</div>



```python
def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

X,X_test=fe_stats(X,X_test)
display(X.head(2))
print(X.shape)
display(X_test.head(2))
print(X_test.shape)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>c_sum</th>
      <th>c_mean</th>
      <th>c_std</th>
      <th>c_kurt</th>
      <th>c_skew</th>
      <th>gc_sum</th>
      <th>gc_mean</th>
      <th>gc_std</th>
      <th>gc_kurt</th>
      <th>gc_skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>1.134849</td>
      <td>0.907687</td>
      <td>-0.416385</td>
      <td>-0.966814</td>
      <td>-0.254723</td>
      <td>-1.017473</td>
      <td>...</td>
      <td>48.445252</td>
      <td>0.484453</td>
      <td>0.729866</td>
      <td>-0.305222</td>
      <td>0.068575</td>
      <td>40.719854</td>
      <td>0.046697</td>
      <td>0.862286</td>
      <td>-0.255588</td>
      <td>-0.009789</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>0.119282</td>
      <td>0.681738</td>
      <td>0.272399</td>
      <td>0.080113</td>
      <td>1.205169</td>
      <td>0.686517</td>
      <td>...</td>
      <td>49.649828</td>
      <td>0.496498</td>
      <td>0.607660</td>
      <td>0.106481</td>
      <td>-0.159728</td>
      <td>50.682229</td>
      <td>0.058122</td>
      <td>0.830287</td>
      <td>-0.204193</td>
      <td>-0.037680</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1579 columns</p>
</div>


    (23814, 1579)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>c_sum</th>
      <th>c_mean</th>
      <th>c_std</th>
      <th>c_kurt</th>
      <th>c_skew</th>
      <th>gc_sum</th>
      <th>gc_mean</th>
      <th>gc_std</th>
      <th>gc_kurt</th>
      <th>gc_skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>-0.755294</td>
      <td>0.214796</td>
      <td>-0.774511</td>
      <td>0.705349</td>
      <td>1.564580</td>
      <td>-0.194968</td>
      <td>...</td>
      <td>-11.302477</td>
      <td>-0.113025</td>
      <td>0.675216</td>
      <td>-0.300185</td>
      <td>0.284487</td>
      <td>-48.858313</td>
      <td>-0.056030</td>
      <td>0.740781</td>
      <td>0.017836</td>
      <td>0.152192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>-0.186773</td>
      <td>0.374643</td>
      <td>1.176402</td>
      <td>-0.652299</td>
      <td>-0.546638</td>
      <td>-0.403447</td>
      <td>...</td>
      <td>-55.379215</td>
      <td>-0.553792</td>
      <td>0.626056</td>
      <td>0.196417</td>
      <td>0.656194</td>
      <td>-87.283151</td>
      <td>-0.100095</td>
      <td>0.925253</td>
      <td>-0.678799</td>
      <td>0.119278</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1579 columns</p>
</div>


    (3982, 1579)
    


```python
from sklearn.cluster import KMeans
def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 239):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

X ,X_test=fe_cluster(X,X_test)
display(X.head(2))
print(X.shape)
display(X_test.head(2))
print(X_test.shape)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>clusters_g_30</th>
      <th>clusters_g_31</th>
      <th>clusters_g_32</th>
      <th>clusters_g_33</th>
      <th>clusters_g_34</th>
      <th>clusters_c_0</th>
      <th>clusters_c_1</th>
      <th>clusters_c_2</th>
      <th>clusters_c_3</th>
      <th>clusters_c_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>1.134849</td>
      <td>0.907687</td>
      <td>-0.416385</td>
      <td>-0.966814</td>
      <td>-0.254723</td>
      <td>-1.017473</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>0.119282</td>
      <td>0.681738</td>
      <td>0.272399</td>
      <td>0.080113</td>
      <td>1.205169</td>
      <td>0.686517</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1619 columns</p>
</div>


    (23814, 1619)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>...</th>
      <th>clusters_g_30</th>
      <th>clusters_g_31</th>
      <th>clusters_g_32</th>
      <th>clusters_g_33</th>
      <th>clusters_g_34</th>
      <th>clusters_c_0</th>
      <th>clusters_c_1</th>
      <th>clusters_c_2</th>
      <th>clusters_c_3</th>
      <th>clusters_c_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>-0.755294</td>
      <td>0.214796</td>
      <td>-0.774511</td>
      <td>0.705349</td>
      <td>1.564580</td>
      <td>-0.194968</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>-0.186773</td>
      <td>0.374643</td>
      <td>1.176402</td>
      <td>-0.652299</td>
      <td>-0.546638</td>
      <td>-0.403447</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1619 columns</p>
</div>


    (3982, 1619)
    


```python
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(0.8)  
data = X.append(X_test)
data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : X.shape[0]]
test_features_transformed = data_transformed[-X_test.shape[0] : ]


X = pd.DataFrame(X[['sig_id','cp_type', 'cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

X = pd.concat([X, pd.DataFrame(train_features_transformed)], axis=1)


X_test = pd.DataFrame(X_test[['sig_id','cp_type', 'cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

X_test = pd.concat([X_test, pd.DataFrame(test_features_transformed)], axis=1)

display(X.head(2))
print(X.shape)
display(X_test.head(2))
print(X_test.shape)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>...</th>
      <th>1034</th>
      <th>1035</th>
      <th>1036</th>
      <th>1037</th>
      <th>1038</th>
      <th>1039</th>
      <th>1040</th>
      <th>1041</th>
      <th>1042</th>
      <th>1043</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>1.134849</td>
      <td>0.907687</td>
      <td>-0.416385</td>
      <td>-0.966814</td>
      <td>-0.254723</td>
      <td>-1.017473</td>
      <td>...</td>
      <td>0.908941</td>
      <td>-1.080358</td>
      <td>-0.253258</td>
      <td>0.115213</td>
      <td>-0.226061</td>
      <td>-7.725398</td>
      <td>48.445252</td>
      <td>-0.305222</td>
      <td>0.068575</td>
      <td>40.719854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>0.119282</td>
      <td>0.681738</td>
      <td>0.272399</td>
      <td>0.080113</td>
      <td>1.205169</td>
      <td>0.686517</td>
      <td>...</td>
      <td>-0.502353</td>
      <td>0.673057</td>
      <td>-0.272862</td>
      <td>0.455034</td>
      <td>-1.085957</td>
      <td>1.032402</td>
      <td>49.649828</td>
      <td>0.106481</td>
      <td>-0.159728</td>
      <td>50.682229</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1048 columns</p>
</div>


    (23814, 1048)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_type</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>...</th>
      <th>1034</th>
      <th>1035</th>
      <th>1036</th>
      <th>1037</th>
      <th>1038</th>
      <th>1039</th>
      <th>1040</th>
      <th>1041</th>
      <th>1042</th>
      <th>1043</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>trt_cp</td>
      <td>1</td>
      <td>0</td>
      <td>-0.755294</td>
      <td>0.214796</td>
      <td>-0.774511</td>
      <td>0.705349</td>
      <td>1.564580</td>
      <td>-0.194968</td>
      <td>...</td>
      <td>0.099395</td>
      <td>-0.800232</td>
      <td>0.523428</td>
      <td>-0.393640</td>
      <td>-0.745169</td>
      <td>-37.555836</td>
      <td>-11.302477</td>
      <td>-0.300185</td>
      <td>0.284487</td>
      <td>-48.858313</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>trt_cp</td>
      <td>3</td>
      <td>0</td>
      <td>-0.186773</td>
      <td>0.374643</td>
      <td>1.176402</td>
      <td>-0.652299</td>
      <td>-0.546638</td>
      <td>-0.403447</td>
      <td>...</td>
      <td>-0.612847</td>
      <td>1.042001</td>
      <td>0.832567</td>
      <td>1.149697</td>
      <td>0.106494</td>
      <td>-31.903936</td>
      <td>-55.379215</td>
      <td>0.196417</td>
      <td>0.656194</td>
      <td>-87.283151</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1048 columns</p>
</div>


    (3982, 1048)
    


```python
y0 = y0[X['cp_type'] == 'trt_cp'].reset_index(drop = True)
y = y[X['cp_type'] == 'trt_cp'].reset_index(drop = True)
X = X[X['cp_type'] == 'trt_cp'].reset_index(drop = True)
X.drop(['cp_type','sig_id'], axis=1, inplace=True)
X_test.drop(['cp_type','sig_id'], axis=1, inplace=True)

print('New data shape', X.shape)
```

    New data shape (21948, 1046)
    

# Training

## Model definition
Here we define our neural network model which consists of several dense, dropout and batchnorm layers. I used different activations after my dense layers. We first train the network on non-scored targets and then transfer the weights to train another model on the scored targets. Smoothing the labels may prevent the network from becoming over-confident and has some sort of regularization effect <a href="https://www.kaggle.com/rahulsd91/moa-label-smoothing">[4] </a>. It seems this method works well here. I used Keras Tuner to tune the hyperparameters. The details are in this notebook <a href="https://www.kaggle.com/sinamhd9/hyperparameter-tuning-with-keras-tuner">[5] </a>


```python
p_min = 0.001
p_max = 0.999
from tensorflow.keras import regularizers

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,p_min,p_max)
    return -K.mean(y_true*K.log(y_pred) + (1-y_true)*K.log(1-y_pred))

def create_model(num_cols, hid_layers, activations, dropout_rate, lr, num_cols_y):
    
    inp1 = tf.keras.layers.Input(shape = (num_cols, ))
    x1 = tf.keras.layers.BatchNormalization()(inp1)

    for i, units in enumerate(hid_layers):
        x1 = tf.keras.layers.Dense(units, activation=activations[i])(x1)
        x1 = tf.keras.layers.Dropout(dropout_rate[i])(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
    
    x1 = tf.keras.layers.Dense(num_cols_y,activation='sigmoid')(x1)
    model = tf.keras.models.Model(inputs= inp1, outputs= x1)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
                 loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001), metrics=logloss)
    
    return model 
    
```


```python
hid_layers = [[[512, 768, 896],[384, 640, 1024],[768,768,896],[512, 384, 1024],
              [512, 640, 640], [640, 896, 1024], [256,640,896],[512,512,768],
              [512, 384, 896],[512,768,768]],
             [[512, 768, 896],[384, 640, 1024],[768,768,896],[512, 384, 1024],
              [512, 640, 640], [640, 896, 1024], [256,640,896],[512,512,768],
              [512, 384, 896],[512,768,768]],
              [[1152, 640, 3072],[896, 1024, 3584],[1920,1024,3712],[896, 1024, 3456],
              [1408, 896, 3456], [1408, 768, 3456], [2176,640,3840],[1664,1280,2688],
              [1792, 768, 2432],[1280,1664,4096]],
            [[896, 1792, 3712],[2048, 1024, 1664],[1664,1408,1920],[896, 1408, 3072],
              [1152, 1152, 3072], [2816, 3072, 3328], [2304,2432,2176],[3968,4096,2816],
              [1920, 1536, 3072],[128,1920,1664]],
         [[896, 1152, 1408],[1920, 768, 2176],[2048,2048,1792],[2304, 1664, 512],
              [768, 384, 512], [640, 1664, 512], [1664,1920,2688],[2432,1664,1536],
              [640, 896, 2432],[1536,2176,2176]]]

dropout_rate = [[[0.65,0.35,0.35],[0.65,0.35,0.45],[0.7,0.4,0.4],[0.65,0.35,0.45],
                [0.65,0.35,0.45],[0.7,0.3,0.45],[0.7,0.35,0.4],[0.7,0.4,0.4],
               [0.7, 0.3, 0.4],[0.65, 0.3, 0.4]],
               [[0.65,0.35,0.35],[0.65,0.35,0.45],[0.7,0.4,0.4],[0.65,0.35,0.45],
                [0.65,0.35,0.45],[0.7,0.3,0.45],[0.7,0.35,0.4],[0.7,0.4,0.4],
               [0.7, 0.3, 0.4],[0.65, 0.3, 0.4]],
                [[0.7,0.55,0.7],[0.7,0.4,0.6],[0.7,0.5,0.55],[0.7,0.55,0.6],
                [0.7,0.5,0.65],[0.7,0.5,0.65],[0.7,0.55,0.7],[0.7,0.5,0.65],
               [0.7, 0.35, 0.6],[0.7, 0.5, 0.7]],
                [[0.7,0.4,0.7],[0.7,0.7,0.7],[0.7,0.7,0.7],[0.7,0.25,0.7],
                [0.7,0.4,0.6],[0.7,0.6,0.7],[0.7,0.5,0.7],[0.7,0.7,0.7],
               [0.7, 0.45, 0.7],[0.7, 0.3, 0.6]],
             [[0.7,0.7,0.45],[0.7,0.7,0.6],[0.7,0.7,0.4],[0.7,0.7,0.55],
                [0.65,0.45,0.4],[0.7,0.35,0.5],[0.7,0.7,0.45],[0.7,0.6,0.6],
               [0.7, 0.7, 0.25],[0.7, 0.7, 0.25]]]

activations = [[['elu', 'swish', 'selu'], ['selu','swish','selu'], ['selu','swish','selu'],['selu','swish','elu'],
                ['selu','swish','elu'],['elu','swish','selu'],
               ['selu','swish','elu'],['selu','elu','selu'],['selu','swish','selu'],
               ['selu','swish','elu']],
               [['elu', 'swish', 'selu'], ['selu','swish','selu'], ['selu','swish','selu'],['selu','swish','elu'],
                ['selu','swish','elu'],['elu','swish','selu'],
               ['selu','swish','elu'],['selu','elu','selu'],['selu','swish','selu'],
               ['selu','swish','elu']],
                [['selu', 'relu', 'swish'], ['selu','relu','swish'], ['selu','relu','swish'],['selu','relu','swish'],
               ['selu','relu','swish'],['selu','relu','swish'],['selu','relu','swish'],['selu','relu','swish'],
               ['selu','elu','swish'],['selu','relu','swish']],
                [['selu', 'elu', 'swish'], ['elu','swish','relu'], ['elu','swish','selu'],['selu','elu','swish'],
               ['selu','elu','swish'],['selu','elu','swish'],['selu','elu','swish'],['selu','selu','swish'],
               ['selu','relu','swish'],['elu','elu','swish']],
            [['selu', 'swish', 'selu'], ['selu','swish','selu'], ['elu','swish','selu'],['selu','swish','selu'],
               ['selu','relu','relu'],['selu','relu','relu'],['selu','swish','elu'],['selu','swish','relu'],
               ['elu','swish','selu'],['elu','swish','swish']]]

lr = 5e-4

feats = np.arange(0,X.shape[1],1)
inp_size = int(np.ceil(1* len(feats)))
res = y.copy()
df_sample.loc[:, y.columns] = 0
res.loc[:, y.columns] = 0

```


```python
# Defining callbacks

def callbacks():
    rlr = ReduceLROnPlateau(monitor = 'val_logloss', factor = 0.2, patience = 3, verbose = 0, 
                                min_delta = 1e-4, min_lr = 1e-6, mode = 'min')
        
    ckp = ModelCheckpoint("model.h5", monitor = 'val_logloss', verbose = 0, 
                              save_best_only = True, mode = 'min')
        
    es = EarlyStopping(monitor = 'val_logloss', min_delta = 1e-5, patience = 10, mode = 'min', 
                           baseline = None, restore_best_weights = True, verbose = 0)
    return rlr, ckp, es
```


```python
def log_loss_metric(y_true, y_pred):
    metrics = []
    for _target in y.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels = [0,1]))
    return np.mean(metrics)
```


## Training and validation
We use Multilabel Stratified KFold with 5 splits which is added in the beginning to the notebook.<br>


```python
test_preds = []
res_preds = []
np.random.seed(seed=42)
n_split = 5
n_top = 10
n_round = 1

for seed in range(n_round):

    split_cols = np.random.choice(feats, inp_size, replace=False)
    res.loc[:, y.columns] = 0
    df_sample.loc[:, y.columns] = 0
    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits = n_split, random_state = seed, shuffle = True).split(X, y)):
        
        start_time = time()
        x_tr = X.astype('float64').values[tr][:, split_cols]
        x_val = X.astype('float64').values[te][:, split_cols]
        y0_tr, y0_val = y0.astype(float).values[tr], y0.astype(float).values[te]
        y_tr, y_val = y.astype(float).values[tr], y.astype(float).values[te]
        x_tt = X_test.astype('float64').values[:, split_cols]
        
        for num in range(n_top):
            model = create_model(inp_size, hid_layers[n][num], activations[n][num], dropout_rate[n][num], lr, y0.shape[1])
            model.fit(x_tr, y0_tr,validation_data=(x_val, y0_val), epochs = 150, batch_size = 128,
                      callbacks = callbacks(), verbose = 0)
            model.load_weights("model.h5")
            model2 = create_model(inp_size, hid_layers[n][num], activations[n][num], dropout_rate[n][num], lr, y.shape[1])
            for i in range(len(model2.layers)-1):
                model2.layers[i].set_weights(model.layers[i].get_weights())

            model2.fit(x_tr, y_tr,validation_data=(x_val, y_val),
                            epochs = 150, batch_size = 128,
                            callbacks = callbacks(), verbose = 0)
                       
            model2.load_weights('model.h5')
        
            df_sample.loc[:, y.columns] += model2.predict(x_tt, batch_size = 128)/(n_split*n_top)
        
            res.loc[te, y.columns] += model2.predict(x_val, batch_size = 128)/(n_top)
        
        oof = log_loss_metric(y.loc[te,y.columns], res.loc[te, y.columns])
        print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}], Seed {seed}, Fold {n}:', oof)

        K.clear_session()
        del model2
        x = gc.collect()

    df_sample.loc[ind_te, y.columns] = 0
    
    test_preds.append(df_sample.copy())
    
    res_preds.append(res.copy())
```

    [07:52], Seed 0, Fold 0: 0.015748320586468988
    [07:53], Seed 0, Fold 1: 0.016120888846880125
    [11:40], Seed 0, Fold 2: 0.015879236699007655
    [15:41], Seed 0, Fold 3: 0.015823347366382818
    [11:37], Seed 0, Fold 4: 0.016090145482597268
    


## Blending

We blend the results of all models using averaging. In previous versions we used optimization suggested by this notebook <a href='https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0'>[6] </a>. It is also good to read this notebook to understand the neural split method. <a href='https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras'>[7] </a> <br>Blending may result in score improvement taking into the effect of all the models with slight differences. 


```python

aa = [1.0]
res2= res.copy()
res2.loc[:, y.columns] = 0
for i in range(n_round):
    res2.loc[:, y.columns] += aa[i] * res_preds[i].loc[:, y.columns]
print(log_loss_metric(y, res2))

```

    0.01593238759425166
    


```python
df_sample.loc[:, y.columns] = 0
for i in range(n_round):
    df_sample.loc[:, y.columns] += aa[i] * test_preds[i].loc[:, y.columns]
df_sample.loc[ind_te, y.columns] = 0
```


```python
display(df_sample.head())
df_sample.to_csv('submission.csv', index=False)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor</th>
      <th>11-beta-hsd1_inhibitor</th>
      <th>acat_inhibitor</th>
      <th>acetylcholine_receptor_agonist</th>
      <th>acetylcholine_receptor_antagonist</th>
      <th>acetylcholinesterase_inhibitor</th>
      <th>adenosine_receptor_agonist</th>
      <th>adenosine_receptor_antagonist</th>
      <th>adenylyl_cyclase_activator</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>0.000941</td>
      <td>0.001573</td>
      <td>0.001703</td>
      <td>0.016756</td>
      <td>0.024704</td>
      <td>0.005291</td>
      <td>0.002009</td>
      <td>0.006073</td>
      <td>0.000219</td>
      <td>...</td>
      <td>0.000976</td>
      <td>0.001611</td>
      <td>0.003962</td>
      <td>0.001375</td>
      <td>0.000585</td>
      <td>0.000865</td>
      <td>0.000556</td>
      <td>0.001719</td>
      <td>0.004368</td>
      <td>0.001348</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>0.000540</td>
      <td>0.001223</td>
      <td>0.001453</td>
      <td>0.001407</td>
      <td>0.001562</td>
      <td>0.002896</td>
      <td>0.005221</td>
      <td>0.015678</td>
      <td>0.009869</td>
      <td>...</td>
      <td>0.000661</td>
      <td>0.001053</td>
      <td>0.003647</td>
      <td>0.000365</td>
      <td>0.010849</td>
      <td>0.000789</td>
      <td>0.008113</td>
      <td>0.000991</td>
      <td>0.002173</td>
      <td>0.002325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_002429b5b</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_00276f245</td>
      <td>0.000863</td>
      <td>0.000690</td>
      <td>0.002314</td>
      <td>0.013570</td>
      <td>0.012499</td>
      <td>0.003969</td>
      <td>0.002822</td>
      <td>0.003990</td>
      <td>0.000225</td>
      <td>...</td>
      <td>0.000784</td>
      <td>0.001640</td>
      <td>0.004469</td>
      <td>0.045273</td>
      <td>0.007407</td>
      <td>0.000675</td>
      <td>0.001533</td>
      <td>0.002025</td>
      <td>0.001620</td>
      <td>0.002118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_0027f1083</td>
      <td>0.001467</td>
      <td>0.001673</td>
      <td>0.001701</td>
      <td>0.016926</td>
      <td>0.025723</td>
      <td>0.003811</td>
      <td>0.006591</td>
      <td>0.002371</td>
      <td>0.000360</td>
      <td>...</td>
      <td>0.000779</td>
      <td>0.000791</td>
      <td>0.003467</td>
      <td>0.001586</td>
      <td>0.001005</td>
      <td>0.000848</td>
      <td>0.000970</td>
      <td>0.001453</td>
      <td>0.000323</td>
      <td>0.001164</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 207 columns</p>
</div>


# Evaluation and Summary

In this project, we first examined the data and performed some explanatory data analysis. We then trained a model using deep neural networks on the non-scored targets, transfered the weights and trained the model on scored targets. We then blended the results of different network architectures and initializations. Let's dive deep more into the data and see which label is contributing the most to the overall loss. 


```python
y_true = y
y_preds = res2
```


```python
losses = []
for i in range(y.shape[1]):
    losses.append(log_loss(y.iloc[:,i], res2.iloc[:,i]))
```


```python
max_loss_ind= np.argmax(losses)
max_loss = np.max(losses)
print("Max loss is", max_loss,'For index', max_loss_ind,'which is',y.iloc[:,max_loss_ind].name)
```

    Max loss is 0.08934922372622961 For index 71 which is cyclooxygenase_inhibitor
    


```python
y_max_loss = y.iloc[:,max_loss_ind]
y_max_loss.value_counts()

sns.countplot(y=y_max_loss,palette='nipy_spectral',orient='h')
plt.show()
```


![png](mechanisms_of_action_moa_tutorial_files/mechanisms_of_action_moa_tutorial_63_0.png)


As we can see label 71 is contributing the most to the loss. As we saw earlier, this target was also the third top in having the most positive labels. Some may think of using imblearn library to address the imbalance problem. <a href="https://www.kaggle.com/sinamhd9/safe-driver-prediction-a-comprehensive-project">[8] </a> However, this may get complicated for a multilabel problem .

# References
I would like to express my gratitude to the authors of these kernels who shared their work.
    
<a href="https://www.kaggle.com/kailex/moa-transfer-recipe-with-smoothing"> [1] MOA: Transfer Recipe with Smoothing</a> <br>
<a href="https://www.kaggle.com/c/lish-moa/discussion/184005"> [2] Competition Insights </a> <br>
<a href='https://www.kaggle.com/kushal1506/moa-pytorch-0-01859-rankgauss-pca-nn?scriptVersionId=44558776' >[3] MoA | Pytorch | 0.01859 | RankGauss | PCA | NN </a> <br> 
<a href="https://www.kaggle.com/rahulsd91/moa-label-smoothing">[4] MoA Label Smoothing  <a/> <br>
<a href="https://www.kaggle.com/sinamhd9/hyperparameter-tuning-with-keras-tuner">[5] Hyperparameter tuning with Keras Tuner </a> <br>
<a href='https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0'>[6] Model Blending Weights Optimisation </a> <br>
<a href='https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras'>[7] Split Neural Network Approach (TF Keras) </a> <br>
<a href='https://www.kaggle.com/sinamhd9/safe-driver-prediction-a-comprehensive-project'>[8] Safe driver prediction: A comprehensive project</a> <br>


```python

```
