# MNDO
Python implementation of MNDO (Multivariate Normal Distribution based Oversampling). 

[Article about this implemention](http://ebooks.iospress.nl/volumearticle/49953)


## Requirements
+ Anaconda / Python 3.6
+ tqdm 4.31.1
+ imbalanced-learn 0.4.3


## Usage
### Preprocessing Keel-datasets
If you use [Keel-datasets](http://sci2s.ugr.es/keel/datasets.php), you can use the following command.

```
python pre_dataset.py dataset_directory
```

+ Preprocessing all files in a directory.
+ Remove unnecessary lines and replace class labels. (Positive class -> 1, Negative class -> 0)
+ Preprocessed data is saved in MNDO/Predataset/xxx.csv

### Over-sampling
Resampled(generated) data is stored in ./pos_data
```
python over-sampling.py data_path
```

### Training
```
python train.py data_path
```

train.py steps:
1. Load data
2. Over-sampling (MNDO, SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN and SMOTE-Tomek Links)
3. Scaling (Normalization or Standardization)
4. Learning (SVM, Decision Tree and k-NN)
5. Predict (Results is saved in MNDO/output/xxx.csv)

If you want to train all files, you can use this script:
```
./run.sh
```

## ToDo
+ [ ] Provide as python library

## Related works
+ Kotaro Ambai, Hamido Fujita, MNDO: Multivariate Normal Distribution Based Over-Sampling for Binary Classification, Volume 303: New Trends in Intelligent Software Methodologies, Tools and Techniques, DOI: [10.3233/978-1-61499-900-3-425](http://ebooks.iospress.nl/volumearticle/49953)
+ [Study on improving prediction accuracy for imbalanced medical data using Multivariate Normal Distribution based Oversampling](http://sotsuron.sd.soft.iwate-pu.ac.jp/images/sotsuron/PDF/0312014015_20180111111148_0312014015.pdf)

## Author
Kotaro Ambai ([baibai25](https://github.com/baibai25))
