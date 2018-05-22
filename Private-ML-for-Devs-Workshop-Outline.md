# [Private] ML for Devs Workshop — Outline
*This document and its contents are confidential — please do not share*
*Author: Louis Dorard © All Rights Reserved*

----------
# Preparatory information

+ML for Devs Workshop — Preparation 

# Introduction
https://docs.google.com/presentation/d/13FCEmezlKd_loPF-7uqjf_t8e-aLxIcxH6YSzB_uK0A/edit?usp=sharing
https://docs.google.com/presentation/d/1yKaZsYVCfb8Nk55XmZkjEDohcFnX5KS2rJHTvxM_s7g/edit?usp=sharing

# Check set-up
[ ] Wifi access
[ ] Use Chrome
[ ] Access [tlk.io](http://tlk.io/ml-humancoders)/INSERT_ROOM_NAME (live chat)
[ ] Access [BigML dashboard](http://bigml.com/dashboard)
## Floyd, Jupyter, Python

Assuming you already went through the set-up instructions in +ML/DL for Devs Workshop — Set-Up Instructions …

[ ] Have you installed `floyd-cli`?
[ ] Send your Floyd username to your instructor / to the chat room (we’ll add you to our Team on Floyd)
[ ] From your terminal, create a new directory on your machine, and link it to our team project:
    mkdir ml-workshop
    cd ml-workshop
    floyd init [INSERT_TEAM_NAME]/ml-workshop
[ ] Add the following notebook to your local `ml-workshop` directory
https://www.dropbox.com/s/kl3b80mbqcj3kh0/Intro-Jupyter.ipynb?dl=0

[ ] Run Jupyter server on Floyd: `floyd run --mode jupyter`. Note that this uploads the contents of your local directory to Floyd.
[ ] Your run/job will appear at https://www.floydhub.com/[INSERT_TEAM_NAME]/projects/dl-workshop
[ ] In the Jupyter server, click on `Intro-Jupyter.ipynb`
[ ] Go through notebook
[ ] Make one obvious change
[ ] Stop the Floyd run/job: `floyd stop [INSERT_JOB_NAME]`
[ ] Go to floydhub.com/[INSERT_TEAM_NAME]/projects/test/jobs/[INSERT_JOB_NAME] and look at Overview, Code, Output, CLI tabs
[ ] Download the output data
[ ] Make sure you can see the change you made to the notebook

**Remarks**

- Floyd is not a code repository. It’s just an “execution environment”.
  - It just gives access to cloud instances, for a limited time.
  - As a by-product, it stores the code that was sent from your machine to the cloud instance when you did `floyd run`, and the final versions of files before you destroyed that instance.
- We get billed as soon as Floyd gives you a job name, until we stop the run/job.
- Code and notebook changes are not automatically downloaded to your machine (but they’re easy to get, even after the cloud instance was destroyed).
- You can use a better CPU by just [adding](https://docs.floydhub.com/guides/run_a_job/#instance-type) `[--cpu2](https://docs.floydhub.com/guides/run_a_job/#instance-type)` [to the run command](https://docs.floydhub.com/guides/run_a_job/#instance-type).
## Not using Floyd?

If you run into any issues with the notebooks, please check the versions of Python and libraries with this code (uses Jupyter ‘magics’):

    %load_ext version_information
    %version_information sklearn
# Creating and inspecting Decision Trees
## Classification

**Theory**

https://docs.google.com/presentation/d/1As3S8dxkhYFXUV9x3I_Rh24rwXb8hcHa6isg3sx-nx8/edit?usp=sharing


**Dataset: iris (classification)**

- 150 inputs: measurements taken on irises
- Output: species (setosa, virginica, or versicolor)
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1525624985555_Irises.jpg)
https://docs.google.com/spreadsheets/d/1BxcpSJWL6NF2DqfRlaZmpkPW3jQJgVIJIqxb2p3rOfo/edit?usp=sharing


**Scikit**

[ ] Add the notebook below to your local `ml-workshop` directory
https://www.dropbox.com/s/kzugiyb2x6jawh4/Decision-Tree.ipynb?dl=0

[ ] Add a text file named `floyd_requirements.txt` containing the following:
    scikit-learn==0.19.1
    version-information
    pandas-profiling
[ ] Run Jupyter server on Floyd

**BigML demo**

[ ] Load data source in BigML [dashboard](http://www.bigml.com/dashboard)
[ ] Check types of features detected (configure source, if needed)
[ ] 1-click dataset
[ ] 1-click model (predicting last field, by default)
[ ] Feature importance
[ ] Create tree with only 2 features

**Exercise: tree inspection**

[ ] Work in pairs
[ ] Replicate what was shown in Demo above
[ ] Interact with BigML tree visualization
[ ] Prediction function: based on the visualization of the tree you’ve built, write function below in Python/pseudo-code/other
    def predict(petal_width, petal_length):
      return output
[ ] Draw decision tree boundaries
![](https://www.dropbox.com/s/65s192ngt2moo7y/iris-scatter-plot.png?raw=1)

## Regression

**Dataset: boston-housing (regression)**

- 506 Inputs: Boston neighborhoods
- Output: median home price
- 13 features including average number of rooms per dwelling, crime rate by town, etc.
- More info [here](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1514912255999_boston-houses.png)
https://docs.google.com/spreadsheets/d/1IcidnMC2kYg0gcJ5iR2r-Y2GqWvoy520hEHDgz9Bi28/edit?usp=sharing


**Exercise**

[ ] Adapt Decision-Tree notebook to boston-housing.csv
[ ] Inspect 1-click regression tree on BigML
# Evaluating models
## Regression

**Theory**

https://docs.google.com/presentation/d/19L10bmVzVqVsDvlDFjeHy1AW0WvpaGYJ462oKS6v2Go/edit?usp=sharing


**Scikit**

https://www.dropbox.com/s/k59ns0j7t1mrduq/Evaluate-1.ipynb?dl=0


**Exercise 1: Compute performance metrics**

[ ] Write a Python function that computes MAE, and apply it at the end of the Evaluate notebook.
[ ] Compute performance metrics, using what's already built in scikit:
    from sklearn import metrics
    metrics.
[ ] Which metric do you prefer out of these? Why? Can you think of other metrics to compute in the context of real-estate?
[ ] Post results to chatroom (along with the names of the metrics you used).

**Exercise 2: Randomness in ML**

[ ] Re-run the Evaluate notebook and see that results are different.
[ ] When calling `train_test_split`, set `random_state` to any value you like. Re-run twice and notice that results stay the same this time
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1513637274909_image.png)

## Classification

**Theory**

https://docs.google.com/presentation/d/1jh5n7Vn1piklYPuqWBbV1VGQOHkDY5TUHsE_NJwqsaI/edit?usp=sharing


**Dataset: give me credit (classification)**

- From [Kaggle](http://kaggle.com/c/GiveMeSomeCredit): “predicting the probability that somebody will experience financial distress in the next two years” 
- 120,270 inputs (originally there were 150,000 inputs but we've removed rows that contained missing values, as they pose problems to scikit)
- Output: SeriousDlqin2yrs
- See Data Dictionary for feature descriptions
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1513542909774_image.png)
https://www.dropbox.com/s/9js7t26yb1r5uu9/Credit%20Data%20Dictionary.xls?dl=0
https://docs.google.com/spreadsheets/d/1CaZSHIekU7XsmkXz5K-Pty2WTVunEEr3InNYTVpm_1w/edit?usp=sharing


**Mounting Floyd datasets**
Our datasets have actually been loaded on the Floyd platform already! You can see them [here](https://www.floydhub.com/louisdorard/datasets/favorites). We can also “mount” the contents of this onto the virtual machine created with a `floyd run`… 

[ ] Stop your current job and launch a new one with this additional option: `--data louisdorard/datasets/favorites/3:/data`

**Exercise: adapt regression code to classification**

[ ] In Evaluate notebook, load the credit dataset instead of Boston housing: `data = pd.read_csv("/data/kaggle-give-me-credit-nomissing.csv", index_col=0)`
[ ] Create a decision tree classifier
[ ] Compute relevant performance metrics
[ ] Post your results to the chatroom

**BigML demo**

![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1513542445717_image.png)

[ ] Create *source*
[ ] Check types of features detected (configure source, if needed)
[ ] Create *dataset* from *source* 
[ ] Split intro training and test *datasets* (from dataset view)
[ ] Build *model* from training *dataset* (from training dataset view)
[ ] Evaluate *model* on test *dataset* (from model view)
[ ] Look at confusion matrix


----------
# Optimizing models
## Tuning model complexity

**Theory**

https://docs.google.com/presentation/d/193y_IOQxtl87eXDhEZJKzYuGIjXHanJZas1QlBp1YQI/edit?usp=sharing


**Exercise: max tree depth**

[ ] Try DecisionTreeRegressor with different `max_depth` values and evaluate on boston-housing
[ ] Which is the best max depth value to use? Post results to chat.


## Random Forests

**Theory**

https://docs.google.com/presentation/d/1FwficrjLCbKZNMup2Gx3jeUw6d3CSU8M7dTi8D-dNFE/edit?usp=sharing


**Exercise with scikit**

[ ] Try RandomForestRegressor with 100 trees, different `max_depth` values, and evaluate on boston-housing
[ ] Which is the best max depth value to use? Post results to chat

**Exercise with BigML**

[ ] Evaluate “1-click ensemble” with BigML
[ ] Post results to chat

**BigML demo**

[ ] Look at ensemble view
[ ] Make predictions on test set
## Thresholding soft-classifiers

**Theory**

https://docs.google.com/presentation/d/1qL8u05jhgXanc0RuqU1zbFtPtHNzZxN2U3qX7Om-Ee8/edit?usp=sharing


**Exercise: credit risk scoring in BigML**

[ ] Look at recall, precision, F-measure
[ ] Look at ROC curve and AUC
[ ] Change “probability threshold” and see metrics change (except … ?)
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1516113071352_BigML_ROC.png)

## Tuning hyper-parameters

**Theory**

https://docs.google.com/presentation/d/1iws94wEYCvjwftnk39m8tiKeqiZWZSXjK3b9nnrCIi8/edit?usp=sharing


**Scikit**

https://www.dropbox.com/s/c9dpeaoyqef9t82/Evaluate-2.ipynb?dl=0
https://www.dropbox.com/s/lkz9o31r7pud6bj/Grid-Search.ipynb?dl=0


**Exercise with scikit**

[ ] Grid search RandomForestRegressor with 100 trees on boston-housing, on a range of `max_depth` values
[ ] Which is the best max depth value to use? Post results to chat
[ ] Do the same with `max_features`
[ ] Add `n_jobs=-1` to the options passed to GridSearchCV and re-run. This should now use all cores available on the CPU and make the search faster!
[ ] Results seem better when using `KFold` with `cross_val_score`... can you guess why?

**Exercise: credit risk scoring**

[ ] Adapt above notebooks for `kaggle-give-me-credit-nomissing.csv` and AUC (see [online doc](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) for possible values of `scoring` parameter)
[ ] Which combination of hyperparameters is best?
[ ] (Optional) Apply to test dataset and submit predictions to Kaggle


----------
# Deploying ML models for Text
## Text pre-processing

**Dataset: StumbleUpon evergreen (classification)**

- 656 inputs: web pages
- Output is 1 when webpage is evergreen, 0 when it’s ephemeral
- From Kaggle: “While some pages we recommend, such as news articles or seasonal recipes, are only relevant for a short period of time, others maintain a timeless quality and can be recommended to users long after they are discovered. A high quality prediction of ‘ephemeral’ or ‘evergreen’ would greatly improve a recommendation system like ours.”
- Mix of categorical and numerical features
- 1 textual feature which is the body of the page (note that this is slightly different from the original Kaggle dataset; rows where the body was empty were removed, so we went down from 725 to 656 inputs)
- More info [here](https://www.kaggle.com/c/stumbleupon)
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1514993559596_stumbleupon_evergreen.jpg)
https://docs.google.com/spreadsheets/d/14YoKJSbovMcV0wtFcvyeLI0Cv8ip3q128KUK3ahAi0Y/edit?usp=sharing


**Scikit**

https://www.dropbox.com/s/sixh7gky36c5hai/Text.ipynb?dl=0

## Text featurization and feature selection

**Scikit**

https://www.dropbox.com/s/035z24lfbb48k0s/Text-Featurization.ipynb?dl=0


**Dataset: hotel reviews (classification)**

- 1600 inputs that have only one feature: text
- Output indicates if review is truthful (“True”) or fake (“False”)
- More info [here](http://myleott.com/op-spam.html)
![](https://d2mxuefqeaa7sj.cloudfront.net/s_10D499B13E25D9F4B41FE3B0A80DA7624F76B549091E6B5D1B115FCEB1C8379A_1514993077357_funny3_3111301a.jpg)
https://docs.google.com/spreadsheets/d/1sd_F6VkZngKdyRP7llaXtUn_uj1JRG5zZq2jGy0yziE/edit?usp=sharing


**Exercise: predict "fakeness" of hotel reviews**
Here we’ll aim at creating a model that detects fake hotel reviews. 

[ ] Load `hotel-reviews.csv` in a new notebook and re-use code from previous notebooks to pre-process text, extract features, and select the 50 best ones
[ ] Evaluate the classifier of your choice with a simple train-test split procedure and the performance measure of your choice


## Querying and inspecting predictive APIs

**BigML demo in Google Sheets**

[ ] Load and split hotel reviews data in BigML
[ ] Create 1-click model from train set
[ ] Download test set from BigML
[ ] Load test set in Google Sheets and rename output column to “ground truth”
[ ] Fill in missing values with BigML plugin
[ ] Add column of errors
[ ] Look at error distribution
[ ] Sort by biggest error
[ ] Look at confidence

**Indico**

[ ] Load Postman collection
https://www.dropbox.com/s/m2wnkeakrk8dc8e/ML_Workshop.postman_collection.json?dl=0

[ ] In Indico > Text Sentiment > Headers, replace {{indicokey}} with the API key found at https://indico.io/dashboard/
[ ] Send request to text sentiment analysis API
## Creating an API with Flask

**Flask code example for Floyd**

    from flask import Flask, request, jsonify
    app = Flask(__name__)
    
    @app.route('/<path:path>', methods=['POST'])
    def predict(path):
        text=request.get_json()['data']
        return jsonify(input=text, score=0.5)
        
    if __name__ == '__main__':
        app.run(host='0.0.0.0')

**Exercise: hotel review fakeness**

[ ] Train a model on all the data available and export it to `model.pkl`
[ ] Create a function that takes in a string and returns the probability of it being fake, based on  `model.pkl` and the `predict_proba()` method in scikit
    def predict_fakeness(text)
      return 0.5
[ ] Run that function as a Python script and test it with “This is a beautiful hotel” (or anything else you think of)
[ ] Save the above Flask code example to a file named `app.py` and adapt it to use your new `predict_fakeness` function
[ ] Add a new line to `floyd_requirements.txt`: 
    flask
[ ] Stop any current Floyd runs/jobs
[ ] Download the new files from the last run (including `model.pkl`) with `floyd data clone [INSERT JOB NAME]/output`  
[ ] Run Floyd in serve mode with `floyd run --mode serve`
[ ] Get the URL where the API is served and paste it in Postman
[ ] Send requests to your new API!


----------
# Feedback
[ ] Fill in feedback form sent to you via email (10’)
# Additional topics
https://docs.google.com/presentation/d/1nY5ffyBNGmF5eRl6Elmk6jxsEWLs5KFL8sSYv0e_iRs/edit?usp=sharing
https://docs.google.com/presentation/d/1YGzjnyFzMjyR8qKjMc93-bZbM-eRUhxYyBqjsYRDxu4/edit?usp=sharing


**Demo BigML:**

[ ] 1-click anomaly detection
[ ] 1-click clustering
[ ] 1-click deep-net
# Resources to go further
![](https://www.dropbox.com/s/wbw7duntdqg1n7u/Screenshot%202018-03-15%2008.57.56.png?raw=1)

## Tools documentation and guides
- [Scikit](http://scikit-learn.org/stable/documentation.html)
  - [Choosing the right estimator](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) (“cheat sheet”)
- [BigML (Classification and Regression)](https://static.bigml.com/pdf/BigML_Classification_and_Regression.pdf)
## Articles, papers, tutorials:
- [10 Things Everyone Should Know About Machine Learning](https://hackernoon.com/10-things-everyone-should-know-about-machine-learning-d2c79ec43201)
- [A few useful things to know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
- [Rules of Machine Learning](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
- [How to deploy ML models in the Cloud](http://www.slideshare.net/AlexCasalboni/how-to-deploy-machine-learning-models-in-the-cloud)
- [Python Data Analysis Tutorials](https://pythonprogramming.net/data-analysis-tutorials/)
- PAPIs proceedings: [2015](http://proceedings.mlr.press/v50/) & [2016](http://proceedings.mlr.press/v67/)
- [Designing great data products](https://www.oreilly.com/ideas/drivetrain-approach-data-products)
- Blog articles: [DataTau](http://datatau.com/) and [Medium](https://medium.com/tag/machine-learning)
- Kaggle “kernels” (see those for the challenges we did, and others!)
## Books:
- [Machine Learning Mastery](http://machinelearningmastery.com/)
- [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)
- [Python Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition)
- [Data Science at the Command Line](http://datascienceatthecommandline.com/)
- Theory! Maths! Stats! Algorithms! Formulae!
  - [Machine Learning Refined](http://mlrefined.wixsite.com/home-page)
  - [Elements of Statistical Learning](http://web.stanford.edu/~hastie/ElemStatLearn/)
## Deep Learning Workshop :)
# Copyright

[Louis Dorard](http://louisdorard.com) © All Rights Reserved

