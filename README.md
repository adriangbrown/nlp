# Disaster Response Pipeline Project

## Project Summary: Application takes in message data sets related to disaster response and attempts to categorize the type of message for faster delegation and dispatching by the appropriate teams. Output is a web app.
Analysis: Data ingested into this app allow the user to take a message and route to the appropriate team based on the terms used in message.
Conclusion: By transforming the message categories into binary format and stripping down the un-needed elements, we are able to get fairly strong precision and recall percentages in each category making this app useful for dispatching support during a disaster.

## File Descriptions
app
  template
    master.html # main page of web app
    go.html # classification result page of web app
  run.py # Flask file that runs app
data
  disaster_categories.csv # data to process
  disaster_messages.csv # data to process
  process_data.py ingests csv data, processes, and saves to sql database
  message_categories.db # database to save clean data to
models
  train_classifier.py extracts data and predicts what category the message should belong to
  classifier.pkl # saved model
README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`
