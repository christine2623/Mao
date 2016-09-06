# Introduction To The Data
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.linear_model import LogisticRegression

admissions = pd.read_table("admissions.data", delim_whitespace=True)

model = LogisticRegression()
model.fit(admissions[["gpa"]], admissions["admit"])
# Use the LogisticRegression method predict to return the label for each observation in the dataset, admissions.
# Assign the returned list to labels.
labels = model.predict(admissions[["gpa"]])
# Add a new column to the admissions Dataframe named predicted_label that contains the values from labels.
admissions["predicted_label"] = labels
# Use the Series method value_counts and the print function to display the distribution of the values in the label column.
print(admissions["predicted_label"].value_counts())
# Use the Dataframe method head and the print function to display the first 5 rows in admissions.
print(admissions.head())




# Accuracy
# The simplest way to determine the effectiveness of a classification model is prediction accuracy.
"""
In logistic regression, recall that the model's output is a probability between 0 and 1.
To decide who gets admitted, we set a threshold and accept all of the students where their computed probability exceeds that threshold.
This threshold is called the discrimination threshold and scikit-learn sets it to 0.5 by default when predicting labels.
If the predicted probability is greater than 0.5, the label for that observation is 1.
If it is instead less than 0.5, the label for that observation is 0.
"""
"""
An accuracy of 1.0 means that the model predicted 100% of admissions correctly for the given discrimination threshold.
An accuracy of 0.2 means that the model predicted 20% of the admissions correctly.
"""

# Rename the admit column from the admissions Dataframe to actual_label so it's more clear which column contains the predicted labels (predicted_label) and which column contains the actual labels (actual_label).
admissions["actual_label"] = admissions["admit"]
# Compare the predicted_label column with the actual_label column.
# Use a double equals sign (==) to compare the 2 Series objects and assign the resulting Series object to matches.
matches = admissions["actual_label"] == admissions["predicted_label"]
# Use conditional filtering to filter admissions to just the rows where matches is True. Assign the resulting Dataframe to correct_predictions.
correct_predictions = admissions[matches]
# Display the first 5 rows in correct_predictions to make sure the values in the predicted_label and actual_label columns are equal.
print(correct_predictions.head())
# Calculate the accuracy and assign the resulting float value to accuracy.
accuracy = float(correct_predictions.shape[0] / admissions.shape[0])
# Display accuracy using the print function.
print(accuracy)

"""Findings:
It looks like the raw accuracy is around 64.5% which is better than randomly guessing the label (which would result in around a 50% accuracy).
Calculating the accuracy of a model on the dataset used for training is a useful initial step just to make sure
the model at least beats randomly assigning a label for each observation.
"""




# Binary Classification Outcomes
"""
The accuracy doesn't tell us how the model performs on data it wasn't trained on.
A model that returns a 100% accuracy when evaluated on it's training set doesn't tell us how well the model works on data it's never seen before
(and wasn't trained on).
Accuracy also doesn't help us discriminate between the different types of outcomes a binary classification model can make.
"""
# principles of evaluating binary classification models by testing our model's effectiveness on the training data.
"""
We can define these outcomes as:

1. True Postive - The model correctly predicted that the student would be admitted.
Said another way, the model predicted that the label would be Positive, and that ended up being True.
In our case, Positive refers to being admitted and maps to the label 1 in the dataset.
For this dataset, a true positive is whenever predicted_label is 1 and actual_label is 1.

2. True Negative - The model correctly predicted that the student would be rejected.
Said another way, the model predicted that the label would be Negative, and that ended up being True.
In our case, Negative refers to being rejected and maps to the label 0 in the dataset.
For this dataset, a true negative is whenever predicted_label is 0 and actual_label is 0.

3. False Positive - The model incorrectly predicted that the student would be admitted even though the student was actually rejected.
Said another way, the model predicted that the label would be Positive, but that was False (the actual label was True).
For this dataset, a false positive is whenever predicted_label is 1 but the actual_label is 0.

4. False Negative - The model incorrectly predicted that the student would be rejected even though the student was actually admitted.
Said another way, the model predicted that the would be Negative, but that was False (the actual value was True).
For this dataset, a false negative is whenever predicted_label is 0 but the actual_label is 1.
"""

# Extract all of the rows where predicted_label and actual_label both equal 1.
# Then, calculate the number of true positives and assign to true_positives.
true_positives_df = admissions[(admissions["actual_label"] == 1) & (admissions["predicted_label"] == 1)]
# Another way: true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = true_positives_df.shape[0]
# Another way: true_positives = len(admissions[true_positive_filter])

# Extract all of the rows where predicted_label and actual_label both equal 0.
# Then, calculate the number of true negatives and assign to true_negatives.
true_negatives_df = admissions[(admissions["actual_label"] == 0) & (admissions["predicted_label"] == 0)]
true_negatives = true_negatives_df.shape[0]
# Use the print function to display both true_positives and true_negatives.
print(true_positives)
print(true_negatives)




# Sensitivity
# Sensitivity or True Positive Rate - The proportion of applicants that were correctly admitted.
# TRP = (True positives)/(True positives + False Negatives)
# this measure helps us answer the question: How effective is this model at identifying positive outcomes?
"""
If the True Positive Rate is low, it means that the model isn't effective at catching positive cases.
For certain problems, high sensitivity is incredibly important.
We want a highly sensitive model that is able to "catch" all of the positive cases.
"""
# Calculate the number of false negatives (where the model predicted rejected but the student was actually admitted) and
# assign to false_negatives.
false_negatives_df = admissions[(admissions["actual_label"] == 1) & (admissions["predicted_label"] == 0)]
false_negatives = false_negatives_df.shape[0]
# Calculate the sensitivity and assign the computed value to sensitivity.
sensitivity = (true_positives)/(true_positives + false_negatives)
print(sensitivity)




# Specificity
# Specificity or True Negative Rate - The proportion of applicants that were correctly rejected
# TNR = (True Negatives)/(False Positives + True Negatives)
# This helps us answer the question: How effective is this model at identifying negative outcomes?
"""
the specificity tells us the proportion of applicants who should be rejected
(actual_label equal to 0, which consists of False Positives + True Negatives) that were correctly rejected (just True Negatives).
A high specificity means that the model is really good at predicting which applicants should be rejected.
"""
# Calculate the number of false positives (where the model predicted admitted but the student was actually rejected) and assign to false_positives.
false_positives_df = admissions[(admissions["actual_label"] == 0) & (admissions["predicted_label"] == 1)]
false_positives = false_positives_df.shape[0]
# Calculate the specificity and assign the computed value to specificity.
specificity = (true_negatives) / (true_negatives + false_positives)
print(specificity)





"""
The important takeaway is the ability to frame the question you want to answer and
working backwards from that to formulate the correct calculation.
"""
"""
If you want to know how well a binary classification model is at catching positive cases,
you should have the intuition to divide the correctly predicted positive cases by all actually positive cases.

There are 2 outcomes associated with an admitted student (positive case), a false negative and a true positive.
Therefore, by dividing the number of true positives by the sum of false negatives and true positives,
you'll have the proportion corresponding to the model's effectiveness of identifying positive cases.
While this proportion is referred to as the sensitivity, the word itself is secondary to the concept
and being able to work backwards to the formula!
"""

