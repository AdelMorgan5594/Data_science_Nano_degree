import sys
import pandas as pd
import matplotlib
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function that loads the data and category from the paths provided then merge them
    input : 
       messages_filepath : The file path of the messages
       categories_filepath : The file path of the categories
       
    output
        df :merged messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages, on ="id")
    return df
    
def clean_data(df):
    
    """
    cleans the dataframe
    
    input : 
        df : merged messages and categories
       
    output
        clean_df : the data frame after being cleaned
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand = True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x.split("-")[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop("categories",axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories,how = "left")
    # drop duplicates
    clean_df= df.drop_duplicates()
    
    return clean_df


def save_data(df, database_filename):
    """
    save the data frame
    
    input : 
        df : Save the clean dataset into a database
       
    output
        none
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass  


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