# MNDO
Multivariate Normal Distribution based Oversampling

## Usage

First download datasets and move to MNDO/Dataset/

    MNDO
    └── Dataset
        └── folder1 
            └── xxx1.dat
            └── yyy1.dat
    └── Predataset
        └── xxx1.csv
        └── yyy1.csv
    └── output
        └── out_xxx.csv
    └── src
        └── multivariate_os.py
        └── predict_data.py
        └── preprocessing.py
    ├── train.py
    └── pre_dataset.py 

### Preprocessing
If you use [Keel-datasets](http://sci2s.ugr.es/keel/datasets.php), you can use the following command.

    $ python pre_dataset.py Dataset/folder_name

+ Preprocessing all files in a directory.
+ Remove unnecessary lines and replace class labels. (Positive class -> 1, Negative class -> 0)
+ Preprocessed data is saved in MNDO/Predataset/xxx.csv


### Training
Results is saved in output/xxx.csv
    
    $ python train.py Predataset/out_xxx.csv

## Related works
- [Study on improving prediction accuracy for imbalanced medical data using Multivariate Normal Distribution based Oversampling](http://sotsuron.sd.soft.iwate-pu.ac.jp/images/sotsuron/PDF/0312014015_20180111111148_0312014015.pdf)

## Author
Kotaro Ambai / [baibai25](https://github.com/baibai25)
