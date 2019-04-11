import os
import numpy as np
from scipy.misc import imread,imresize
from urllib.request import urlretrieve

def load_notmnist(path=".", letters='ABCDEFGHIJ',
                  img_shape=(28,28),test_size=0.25,one_hot=False):

    root = os.path.join(path, "notMNIST_small")
    
    # download data if it's missing. If you have any problems, go to the urls and load it manually.
    if not os.path.exists(root):
        print("Downloading data...")
        urlretrieve(
            "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz",
            "notMNIST_small.tar.gz")
        
        print("Extracting ...")
        import tarfile
        with tarfile.open("notMNIST_small.tar.gz", "r:gz") as tar:
            tar.extractall(path=path)

    data,labels = [],[]
    print("Parsing...")
    for letter in sorted(os.listdir(root)):
        if letter not in letters: continue
        for img_name in sorted(os.listdir(os.path.join(root, letter))):
            img_path = os.path.join(root, letter, img_name)
            try:
                data.append(imresize(imread(img_path), img_shape))
                labels.append(letter,)
            except:
                print("found broken img: %s [it's ok if <10 images are broken]" % img_path)

    data = np.stack(data)[:,None].astype('float32')
    data = (data - np.mean(data)) / np.std(data)

    #convert classes to ints
    letter_to_i = {l:i for i,l in enumerate(letters)}
    labels = np.array(list(map(letter_to_i.get, labels)))

    if one_hot:
        labels = (np.arange(np.max(labels) + 1)[None,:] == labels[:, None]).astype('float32')

    #split into train/test
    np.random.seed(666)
    permutation = np.arange(len(data))
    np.random.shuffle(permutation)
    
    data = data[permutation]
    labels = labels[permutation]
    
    n_train_samples = int(round(len(data) * (1.0 - test_size)))
    X_train, X_test = data[:n_train_samples], data[n_train_samples:]
    y_train, y_test = labels[:n_train_samples], labels[n_train_samples:]

    return X_train, y_train, X_test, y_test

