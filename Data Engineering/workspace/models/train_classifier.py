# import libraries
import sys
import pandas as pd
import re
import pickle
import warnings
warnings.filterwarnings("ignore")
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    """
    A function to load the saved data 
    input:
        database_filepath: a  path of the database
    output:
        X : message list
        Y : category columns (Target)
        category_names : a list of the category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df["message"]
    Y = df.iloc[: ,4:]
    category_names = Y.columns.tolist()
    return X,Y,category_names
    pass


def tokenize(text):
    """
    a function that takes text and return a clean token
    input : text
    output : a clean token
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    A pipeline with adaboostclassifier tfidf transformer
    """
    
    pipeline_adaboost = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=70, learning_rate = 0.5)))])
    
    return pipeline_adaboost



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Report the f1 score, precision and recall for each output category of the dataset 
    input
        model: the model we want to evaluate
        Xtest : the messages we want to evaluate our model on 
        Y_test :  true label we will compare our model results with
        category_names : the names of the categories
    output
        None
    """
    y_pred = model.predict(X_test)
    
    #category_names = list(df.columns[4:])

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
       


def save_model(model, model_filepath):
    """
    Export the model as a pickle file
    input
        model : the model to be saved
        model_filepath : the path we want to store in 
    output
        None
        
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()