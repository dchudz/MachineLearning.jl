print("\nStarting....") 
using MachineLearning
using Distributions

srand(1)
ncols = 3
ntrain = 20
ntest = 10
# coef = rand(TDist(3), ncols)
coef = [0,5,10]
x = randn(ntrain + ntest, ncols)
y = x*coef
train_indices = 1:ntrain
test_indices = (ntrain+1):(ntrain+ntest)
x_train = x[train_indices,:]
x_test = x[test_indices,:]
y_train = y[train_indices]
y_test = y[test_indices];
rf = fit(x_train, y_train, regression_forest_options(num_trees=100))
print(rf)
println("")

test_index = 1

@printf("\n\nConsidering test row index: %d", test_index)
@printf("\n\nFeature values are:\n%s", x_test[test_index,:])
@printf("\n\nPrediction is %s", predict(rf, x_test[1,:]))

row_importance, feature_importance = row_feature_importance(rf, squeeze(x_test[1, :], 1));
@printf("\n\nImportance of training rows in driving this prediction:\n%s", row_importance)

print("\n\n")
train_indices = [13,17,18]
for train_index in train_indices
	@printf("\nFeature importances for train index %d is:\n%s", train_index, feature_importance[train_index])
end

@printf("Feature values for these train indices are:\n%s", x_train[[13,17,18],:])
