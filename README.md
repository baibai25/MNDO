# MNDO
Multivariate Normal Distribution based Oversampling

## Usage

First download [Keel datasets](http://sci2s.ugr.es/keel/datasets.php) and move to /MNDO/Dataset/

    MNDO
    └── Dataset
        └── xxx.dat
        └── yyy.dat
    └── Predataset
        └── out_xxx.csv
        └── out_yyy.csv
    ├── train.ipynb
    ├── pre_dataset.py
    └── Multivariate_os.py 

Preprocessing:
Preprocessed data is saved in /MNDO/Predataset/out_xxx.csv .

    $ python pre_dataset.py xxx.dat

- Positive class -> 1
- Negative class -> 0

## Related works
- [Study on improving prediction accuracy for imbalanced medical data using Multivariate Normal Distribution based Oversampling](http://sotsuron.sd.soft.iwate-pu.ac.jp/images/sotsuron/PDF/0312014015_20180111111148_0312014015.pdf)

## Author
Kotaro Ambai / [baibai25](https://github.com/baibai25)
