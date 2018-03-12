import glob, os
import pandas as pd

def get_name(files):
    name = []
    for i in range(len(files)):
        basename = os.path.basename(files[i])
        split = os.path.splitext(basename)
        name.append(split[0])
    return name

if __name__ == '__main__':
    files = glob.glob('output/*.csv')
    name = get_name(files)
    ind = ['sm', 'b1', 'b2', 'enn', 'tom', 'ada', 'mnd']
    col = ['os', 'sen(macro)', 'sen(micro)', 'spe(macro)', 'spe(micro)', 'geo(macro)', 'geo(micro)', 'AUC']
    
    for i in range(len(ind)):
        svm = pd.DataFrame(index=[], columns=col)
        tree = pd.DataFrame(index=[], columns=col)
        knn = pd.DataFrame(index=[], columns=col)

        for j in range(len(files)):
            data = pd.read_csv(files[j])
            a = data[data['os'] == ind[i]]

            svm = pd.concat([svm, a.iloc[[0]]], ignore_index=True)
            tree = pd.concat([tree, a.iloc[[1]]], ignore_index=True)
            knn = pd.concat([knn, a.iloc[[2]]], ignore_index=True)
        
        df = pd.concat([svm, tree, knn], axis=1)
        df.index = name
        df.drop('os', axis=1, inplace=True)
        path = 'output/all/{}.csv'.format(ind[i])
        df.to_csv(path)
