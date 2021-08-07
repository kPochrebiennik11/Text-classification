import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


def read_data(author):
    main_folder = "scale_data/scaledata"
    # file_names = ["id", "rating", "subj"]
    file_names = ["id", "label.4class", "subj"]
    file_paths = [main_folder + "/" + author + "/" + name + "." + author for name in file_names]
    # print(file_paths)

    with open(file_paths[0]) as f:
        ids = f.readlines()
    ids = [x.strip() for x in ids]

    with open(file_paths[1]) as f:
        ratings = f.readlines()
    ratings = [x.strip() for x in ratings]

    with open(file_paths[2]) as f:
        subjects = f.readlines()
    subjects = [x.strip() for x in subjects]

    print(ids)
    print(ratings)

    print(len(ids))
    print(len(ratings))
    # print(len(subjects))
    return ids, ratings, subjects

def prepare_data():
    ds_ids, ds_ratings, ds_subjects = read_data("Dennis+Schwartz")
    jb_ids, jb_ratings, jb_subjects = read_data("James+Berardinelli")
    scr_ids, scr_ratings, scr_subjects = read_data("Scott+Renshaw")
    str_ids, str_ratings, str_subjects = read_data("Steve+Rhodes")

    ids = ds_ids + jb_ids + scr_ids + str_ids
    subjects = ds_subjects + jb_subjects + scr_subjects + str_subjects
    ratings = ds_ratings + jb_ratings + scr_ratings + str_ratings
    print(len(ids))
    print(len(ratings))
    print(len(subjects))
    combine_into_csv(ids, ratings, subjects)

def combine_into_csv(ids, ratings, subjects):
    with open('review_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')                                                                     #, quoting=csv.QUOTE_MINIMAL

        data = list(zip(ids, ratings, subjects))
        writer.writerow(["ID", "Rating", "Subject"])
        for row in data:
            row = list(row)
            writer.writerow(row)

def explore_csv():
    dat2 = pd.read_csv("review_data.csv")

    print()
    print("DATA SHAPE")
    print(dat2.shape)
    print()

    print("DATA FEW FIRST ROWS")
    print(dat2)
    print()

    print("DATA DESCRIPTION")
    print(dat2.describe())
    print()

    print("NUMBER OF NULL VALUES")
    print(dat2.isnull().sum())
    print()

    print("DATA INFO")
    dat2.info()
    print()

    print("RATING VALUE COUNTS")
    print(dat2['Rating'].value_counts())
    print()
    # print(dat2['Subject'])

    print("Subject lengths info")
    print(dat2['Subject'].str.len().describe())
    print("Subject lengths - TOP 20")
    print(dat2['Subject'].str.len().value_counts().nlargest(20))
    print("Subject lengths - BOTTOM 20")
    print(dat2['Subject'].str.len().value_counts().nsmallest(20))
    print()

    words = []
    for subject in dat2['Subject']:
        words += subject.split()

    words = pd.DataFrame(words, columns=['word'])
    word_counts = words['word'].value_counts()

    print("Single word counts")
    print(word_counts)
    print("Single word counts - TOP 50")
    print(word_counts.nlargest(50))
    print("Single word counts - BOTTOM 20")
    print(word_counts.nsmallest(20))
    print()

    # print(word_counts[lambda x: x>5000])

    fig = plt.figure(figsize=(19.2, 10.8))
    dat2.groupby('Rating').ID.count().plot.bar(ylim=0)
    plt.show()

    fig = plt.figure(figsize=(19.2, 10.8))
    word_counts[lambda x: x > 3000].plot.bar(ylim=0)
    plt.show()

    # fig = plt.figure(figsize=(19.2, 10.8))
    # word_counts[lambda x: x < 2].plot.bar(ylim=0)
    # plt.show()

    vectorizer = CountVectorizer()                                                                                      #Change into token count matrix
    X = vectorizer.fit_transform(dat2['Subject'])
    print(vectorizer.get_feature_names())
    print(X.toarray())
    print(X.toarray()[0].sum())

def test():
    def check_model(model):
        model_fit = model.fit(X_train_tfidf.todense(), y_train)
        model_y_pred = model_fit.predict(tfidf_transformer.transform(tfidf_vect.transform(X_test)).todense())

        conf_mat = confusion_matrix(y_test, model_y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=rating_df.Rating.values, yticklabels=rating_df.Rating.values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        print(metrics.classification_report(y_test, model_y_pred))


    df = pd.read_csv("review_data.csv")
    print("DATA FEW FIRST ROWS")
    print(df)
    print()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Subject).toarray()
    labels = df.Rating
    print(features.shape)

    rating_df = df[['Rating']].drop_duplicates().sort_values('Rating')

    rating_df["Meaning"] = ['Bad', 'Slightly bad', 'Slightly good', 'Good']
    print(rating_df.values)
    rating_to_id = dict(rating_df.values)
    # print(rating_to_id)

    N = 5
    for rating_id, label in sorted(rating_to_id.items()):
        features_chi2 = chi2(features, labels == rating_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(label))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    X_train, X_test, y_train, y_test = train_test_split(df['Subject'], df['Rating'], random_state=0)
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(X_train)

    # tfidf_vect = TfidfVectorizer(max_df=0.5, stop_words='english', use_idf=True)
    tfidf_vect = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    X_train_counts = tfidf_vect.fit_transform(X_train)
    print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)




    print("########## MultinomialNB ##########")

    mnb = MultinomialNB()
    check_model(mnb)

    print("########## GaussianNB ##########")

    gnb = GaussianNB()
    check_model(gnb)

    print("########## Complement NB ##########")

    cnb = ComplementNB()
    check_model(cnb)

    print("########## LinearSVC ##########")

    lsvc = LinearSVC()
    check_model(lsvc)

    print("########## LinearSVC with grid search parameters ##########")

    lsvc_2 = LinearSVC()
    parameters_lsvc = {'C': [1, 10]}
    clf_lsvc = GridSearchCV(lsvc_2, parameters_lsvc, scoring='accuracy', verbose=10, n_jobs=5, cv=10)
    clf_lsvc.fit(X_train_tfidf.todense(), y_train)
    print(sorted(clf_lsvc.cv_results_.keys()))
    check_model(lsvc_2)

    print("########## SVC ##########")

    svc = SVC()
    # parameters_svc = {'kernel': ('linear', 'poly', 'sigmoid'), 'C': [1, 10]}
    # clf_svc = GridSearchCV(svc, parameters_svc, scoring='accuracy', verbose=10, n_jobs=5)
    # clf_svc.fit(X_train_tfidf.todense(), y_train)
    # print(sorted(clf_svc.cv_results_.keys()))
    check_model(svc)



if __name__ == '__main__':
    # prepare_data()
    # explore_csv()
    test()

    # subjects_new = ['This movie is really bad, i hated it.', 'I loved the movie.', X_train[3600]]
    # X_new_counts = count_vect.transform(subjects_new)
    # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    # predicted = clf.predict(X_new_tfidf.todense())
    # for subj, category in zip(subjects_new, predicted):
    #     print('%r => %s' % (subj, category))

    # print(X_train_tfidf)