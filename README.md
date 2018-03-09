# MNDO
Multivariate Normal Distribution based Oversampling

## Usage

First download datasets and move to /MNDO/Dataset/

    MNDO
    └── Dataset
        └── xxx.dat
        └── yyy.dat
    └── Predataset
        └── out_xxx.csv
        └── out_yyy.csv
    └── output
        └── svm_out_xxx.csv
        └── knn_out_yyy.csv        
    ├── train.ipynb
    ├── train.py
    ├── pre_dataset.py
    └── multivariate_os.py 

### Preprocessing
If you use [Keel-datasets](http://sci2s.ugr.es/keel/datasets.php), you can use the following command.
+ Remove unnecessary lines and replace.
+ Replace class labels. (Positive class -> 1, Negative class -> 0)
+ Preprocessed data is saved in MNDO/Predataset/out_xxx.csv

    $ python pre_dataset.py xxx.dat

###Training
[classification_report_imbalanced](http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.metrics.classification_report_imbalanced.html) is saved in output/svm_out_xxx.csv
    
    $ python train.py Predataset/out_xxx.csv

## Related works
- [Study on improving prediction accuracy for imbalanced medical data using Multivariate Normal Distribution based Oversampling](http://sotsuron.sd.soft.iwate-pu.ac.jp/images/sotsuron/PDF/0312014015_20180111111148_0312014015.pdf)

## Author
Kotaro Ambai / [baibai25](https://github.com/baibai25)
