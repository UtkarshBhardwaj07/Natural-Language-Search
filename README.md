# Natural Language Search

This project demonstrates the use of Transformer Model.

The user can input a text and get the following:

1) Next word prediction: This predicts the next word in the sequence entered by the user. The user can also specify how many predictions he/she needs.

2) Similar Questions Retrieval: This displays the similar questions for a query entered by the user. The user can specify the number of similar questions he/she needs.

3) Question-Answer: This displays the relevant answers to the corresponding question entered by the user. the user can specify the number of answers he/she needs.


There are two implementations - using STREAMLIT library and other one is using FastAPI.
Both implementations are available in this repo.


Please note that the FastAPI implementation does not contain the Next Word Prediction functionality and also the input is same for both the APIs i.e. Similar Questions Retrieval
and Question-Answer. Therefore, when using STREAMLIT, user can select a particular functionality whereas in FastAPI, the output always includes both the Similar Questions as well as Answers.



For STREAMLIT:

Open Anaconda Terminal and go to the directory where app.py file is stored and run the following command:

streamlit run app.py




For FastAPI:

Open Anaconda Terminal and go to the directory where FastAPI.py file is stored and run the following command:

uvicorn FastAPI:app --reload





