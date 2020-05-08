# Disaster Response Pipeline Project

### Project Motivation

In **Data Engineering** section In the **Data Science NanoDegree**,I have learned and built my data engineering skills to expand your opportunities and potential as a data scientist. In this project, I applied these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, I Found a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Below is a screenshots of the web app.
![](Images/newplot.png)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/. However this did not work and I made the following steps as provided by udacity 

**Udacity workspace**
- Run your app with python run.py command
- Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
- Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
- Press enter 

**Local Machine**
- Once your app is running (python run.py)
- Go to http://localhost:3001 and the app will now run

### Result examples
The first sentence **(We are more than 50 people sleeping in the street. Please help us find home,food)**

![](Images/First_sentence.png)

The second sentence **(In 2020 there were a lot of disasters occurring. It started by the amazon forests fires, Earthquake, floods)**

![](Images/Second_sentence.png)


