# Introduction To Validation
"""
So far, we've been evaluating accuracy of trained models on the data the model was trained on.
While this is an essential first step, this doesn't tell us much about how well the model does on data it's never seen before.
In machine learning, we want to use training data, which is historical and contains the labelled outcomes for each observation,
to build a classifier that will return predicted labels for new, unlabelled data.
If we only evaluate a classifier's effectiveness on the data it was trained on, we can run into overfitting,
where the classifier only performs well on the training but doesn't generalize to future data.
"""
"""
To test a classifier's generalizability, or its ability to provide accurate predictions on data it wasn't trained on,
we use cross-validation techniques. Cross-validation involves splitting historical data into:

a training set -- which we use to train the classifer,
a test set -- which we use to evaluate the classifier's effectiveness using various measures.
"""
# Cross-validation is an important step that should be utilized after training any kind of machine learning model.

import pandas as pd
from sklearn.linear_model import LogisticRegression

admissions = pd.read_table("admissions.data", delim_whitespace=True)
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

print(admissions.head())





# Holdout Validation
"""
holdout validation, which involves:

1. randomly splitting our dataset into a training data and a test set,
2. fitting the model using the training set,
3. making predictions on the test set.
"""
# We'll randomly select 80% of the observations in the admissions Dataframe as the training set and the remaining 20% as the test set.
"""
To split the data randomly into a training and a test set, we'll:

use the NumPy rand.permutation function to return a list containing index values in random order,
return a new Dataframe in that list's order,
select the first 80% of the rows as the training set,
select the last 20% of the rows as the test set.
"""
# Use the NumPy rand.permutation function to randomize the index for the admissions Dataframe.
from numpy.random import permutation
# Use the loc[] method on the admissions Dataframe to return a new Dataframe in the randomized order. Assign this Dataframe to shuffled_admissions.
shuffled_admissions = admissions.loc[permutation(admissions.index)]
# Another way: shuffled_index = np.random.permutation(admissions.index), then: shuffled_admissions = admissions.loc[shuffled_index]
# Select rows 0 to 514 (including row 514) from shuffled_admissions and assign to train.
train = shuffled_admissions[:515]
# Select the remaining rows and assign to test.
test = shuffled_admissions[515:]
# Finally, display the first 5 rows in shuffled_admissions.
print(shuffled_admissions.head())




# Accuracy
"""
train a logistic regression model on just the training set,
use the model to predict labels for the test set,
evaluate the accuracy of the predicted labels for the test set.
Recall that accuracy helps us answer the question:
What fraction of the predictions were correct (actual label matched predicted label)?
"""
# Train a logistic regression model using the gpa column from the train Dataframe.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
logistic_model = model.fit(train[["gpa"]], train["actual_label"])
# Use the LogisticRegression method predict to return the predicted labels for the gpa column from the test Dataframe. Assign the resulting list of labels to the predicted_label column in the test Dataframe.
label = model.predict(test[["gpa"]])
test["predicted_label"] = label
# Calculate the accuracy of the predictions by dividing the number of rows where actual_label matches predicted_label by the total number of rows in the test set.
match = test["predicted_label"] == test["actual_label"]
correct_match = test[match]
# Assign the accuracy value to accuracy and display it using the print function.
accuracy = len(correct_match)/test.shape[0]
# Another way: accuracy2 = correct_match.shape[0]/test.shape[0]
print(accuracy)
# print(accuracy2)





# Sensitivity And Specificity
"""
Looks like the prediction accuracy is about 62%,
which isn't too far off from the accuracy value we computed in the previous mission of 64.6%.
Interestingly enough, the accuracy was higher when we used less data to train the model and when it was evaluated on unseen observations.
This aligns with the fact that computing accuracy on just the data the model was trained on just provides a baseline estimate.
"""
# Calculate the sensitivity value for the predictions on the test set and assign to sensitivity.
true_positives = (test["actual_label"] == 1) & (test["predicted_label"] == 1)
num_true_positives = len(test[true_positives])
false_negatives = (test["actual_label"] == 1) & (test["predicted_label"] == 0)
num_false_negatives = len(test[false_negatives])
sensitivity = (num_true_positives) / (num_false_negatives + num_true_positives)
# Calculate the specificity value for the predictions on the test set and assign to specificity.
false_positives = (test["actual_label"] == 0) & (test["predicted_label"] == 1)
num_false_positives = len(test[false_positives])
true_negatives = (test["actual_label"] == 0) & (test["predicted_label"] == 0)
num_true_negatives = len(test[true_negatives])
specificity = (num_true_negatives) / (num_false_positives + num_true_negatives)
# Display both values using the print function.
print(sensitivity)
print(specificity)






# False Positive Rate
"""
If the probability value is larger than 50%, the predicted label is 1 and if it's less than 50%, the predictd label is 0.
For most problems, however, 50% is not the optimal discrimination threshold.
We need a way to vary the threshold and compute the measures at each threshold.
Then, depending on the measure we want to optimize, we can find the appropriate threshold to use for predictions.
"""
"""
The 2 common measures that are computed for each discrimination threshold are the False Positive Rate (or fall-out)
and the True Positive Rate (or sensitivity).
Fall-out or False Positive Rate - The proportion of applicants who should have been rejected (actual_label equals 0)
but were instead admitted (predicted_label equals 1).
True Positive Rate: The proportion of students that were admitted that should have been admitted.
False Positive Rate: The proportion of students that were accepted that should have been rejected.
FPR = (False Positives)/(True Negatives + False Positives)
"""





# ROC Curve
"""
We can vary the discrimination threshold and calculate the TPR and FPR for each value.
This is called an ROC curve, which stands for reciever operator curve,
and it allows us to understand a classification model's performance as the discrimination threshold is varied.
"""
"""
To calculate the TPR and FPR values at each discrimination threshold, we can use the scikit-learn roc_curve function.
This function will calculate the false positive rate and true positive rate for varying discrimination thresholds until both reach 0%.

This function takes 2 required parameters:
y_true: list of the true labels for the observations,
y_score: list of the model's probability scores for those observations.
fpr, tpr, thresholds = metrics.roc_curve(labels, probabilities)
"""
import matplotlib.pyplot as plt
# Import the relevant scikit-learn package you need to calculate the ROC curve.
from sklearn.metrics import roc_curve
# Use the model to return predicted probabilities for the test set.
pred_prob = model.predict_proba(test[["gpa"]])
# Use the roc_curve function to return the FPR and TPR values for different thresholds.
fpr, tpr, thresholds = roc_curve(test["actual_label"], pred_prob[:,1])
# Create and display a line plot with: the FPR values on the x-axis and the TPR values on the y-axis.
plt.plot(fpr, tpr)
plt.show()




# Area Under The Curve
"""
When looking at an ROC curve, you want to keep an eye on how the 2 measures trade off and
select an appropriate threshold based on your priorities.
Given that the university accepts very few students and rejects most of them,
it's probably more concerned with a higher True Positive Rate than a low False Positive Rate.
The university benefits the most if it does a wonderful job admitting a select number of students
that deserve to be admitted than focusing aggressively on accurately rejecting students.
"""
"""area under the curve or AUC for short:
 The AUC describes the probability that the classifier will rank a random positive observation higher than a random negative observation.
 Since randomly guessing converges to a probability of 0.5, the higher the AUC the more accurate the model seems to be.
 """
# To calculate the AUC, we can use the scikit-learn function roc_auc_score,
# which takes the same parameters as the roc_curve function and returns a single float value corresponding to the AUC.
from sklearn.metrics import roc_auc_score
# Calculate the AUC score for our model on the training set and assign to auc_score.
auc_score = roc_auc_score(test[["actual_label"]], pred_prob[:,1])
# Use the print function to display auc_score.
print(auc_score)

""" findings:
With an AUC score of about 77.8%, our model does a little bit better than 50%, which would correspond to randomly guessing,
but not as high as the university may like.
This could imply that using just one feature in our model, GPA, to predict admissions isn't enough
"""
# the important takeaway is that no single measure will tell us if we want to use a specific model or not.


