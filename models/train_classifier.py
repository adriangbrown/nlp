# import libraries
import sys
import pandas as pd
import re
import nltk
import numpy as np
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sqlalchemy import create_engine, text
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def load_data(database_filepath):
    '''Read in data from sql database
    Inputs: 
    database_filepath:  database location

    Returns:
    X: Input variables
    y: output variables
    category_names:  list of category names of output variables
    '''

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    query = 'SELECT * from message_categories'
    df = pd.read_sql_query(sql=text(query), con=engine.connect())
    #df = pd.read_sql_table('message_categories', conn=engine.connect())
    
    #create dummy value for child_alone
    df.loc[0, 'child_alone'] = 1
    # replace 2 value in related with 1
    df.loc[df['related'] == 2, 'related'] = 1
    
    X = df.message
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''Ingest text and create a list of words that have been lemmatized, stripped of whitespace
    and lowered in case
   
    Input:
    text:  user inputted text
    
    Returns:
    clean_tokens:  list of cleaned up words
    '''

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''Build ML model with various parameters

    Returns:
    model:  model with a Pipeline component
    '''

    clf = RandomForestClassifier()
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
    'clf__estimator__min_samples_split': [2,5]
    }
    
    model = GridSearchCV(model, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates output model effectiveness with a classification report
  
    Input:
    model:  model to use for prediction
    X_test:  Input test data
    Y_test:  Output test data
    category_names:  List of output data category names
    '''

    Y_pred = model.predict(X_test)
    Y_test = np.array(Y_test)
    
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''Save model to pickle file
    Input
    model:  target model
    model_filepath:  directory to park model pickle file
    '''

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
