import pickle

class Classifier:

    def __init__(self):

        self.moviedir = os.getcwd() + '/txt_sentoken'

    def Training(self):

        # loading all files.
        self.movie = load_files(self.moviedir, shuffle=True)


        # Split data into training and test sets
        docs_train, docs_test, y_train, y_test = train_test_split(self.movie.data, self.movie.target,
                                                                  test_size = 0.20, random_state = 12)

        # initialize CountVectorizer
        self.movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=5000)

        # fit and tranform using training text
        docs_train_counts = self.movieVzer.fit_transform(docs_train)


        # Convert raw frequency counts into TF-IDF values
        self.movieTfmer = TfidfTransformer()
        docs_train_tfidf = self.movieTfmer.fit_transform(docs_train_counts)

        # Using the fitted vectorizer and transformer, tranform the test data
        docs_test_counts = self.movieVzer.transform(docs_test)
        docs_test_tfidf = self.movieTfmer.transform(docs_test_counts)

        # Now ready to build a classifier.
        # We will use Multinominal Naive Bayes as our model


        # Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
        self.clf = MultinomialNB()
        self.clf.fit(docs_train_tfidf, y_train)


        # save the model
        filename = 'finalized_model.pkl'
        pickle.dump(self.clf, open(filename, 'wb'))

        # Predict the Test set results, find accuracy
        y_pred = self.clf.predict(docs_test_tfidf)

        # Accuracy
        print(sklearn.metrics.accuracy_score(y_test, y_pred))

        self.Categorize()

    def Categorize(self):
        # very short and fake movie reviews
        reviews_new = ['This movie was excellent', 'Absolute joy ride', 'It is pretty good',
                      'This was certainly a movie', 'I fell asleep halfway through',
                      "We can't wait for the sequel!!", 'I cannot recommend this highly enough', 'What the hell is this shit?']

        reviews_new_counts = self.movieVzer.transform(reviews_new)         # turn text into count vector
        reviews_new_tfidf = self.movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector


        # have classifier make a prediction
        pred = self.clf.predict(reviews_new_tfidf)

        # print out results
        for review, category in zip(reviews_new, pred):
            print('%r => %s' % (review, self.movie.target_names[category]))