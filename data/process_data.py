# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Load two two csv data sets and merge
    Keyword arguments
    messages_filepath:  path to messages csv
    categories_filepath:  path to categories csv
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print(messages.head())
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    print(categories.head())
    # merge datasets
    df = messages.merge(categories, how='inner', left_on='id', right_on='id')
    print(df.head())
    return df

def clean_data(df):
    '''Creates binary category columns in dataset
    Keyword arguments
    df:  target dataframe to clean
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    print(categories.head())
    category_colnames = [item[0] for item in list(categories.iloc[0].str.split(pat='-'))]
    print(category_colnames)
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str.get(1)
    
        # convert column from string to numeric
        categories[column] = categories[column].fillna(0).astype(int)
    print(categories.head())
    
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    print(df.head())
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=True, axis=1)
    print(df.head())
    
    # check number of duplicates
    print(len(df[df.duplicated()]))
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    print(len(df[df.duplicated()]))
    return df


def save_data(df, database_filename):
    '''Store dataframe in SQL database table
    Keyword arguments
    df:  source dataframe
    database_filename:  target database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_categories', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()