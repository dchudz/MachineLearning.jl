using MachineLearning
using DataStructures
dict_mean(dict) = Dict([(key, dict[key]/sum(values(dict))) for key in keys(dict)])
function row_feature_importance(tree::AbstractRegressionTree, sample::Vector{Float64})
    feature_usages = DefaultDict(0)
    node = tree.tree.root
    while typeof(node)==DecisionBranch
        feature_usages[node.feature] += 1
        if sample[node.feature]<=node.value
            node=node.left
        else
            node=node.right
        end
    end
    row_usages = countmap(node.train_row_indices)
    dict_mean(row_usages), dict_mean(feature_usages)
end

function row_feature_importance(rf::RegressionForest, sample::Vector{Float64})
    # Keys will be important training indices
    # Values will hold the sums of their importances across trees
    row_importance_sums = DefaultDict(0) 

    # Keys will be important training indices. 
    # Values will be DefaultDicts whose keys are features, values are sums of the importance of that feature to that training index being important
    feature_importance_sums = DefaultDict(Int, DefaultDict, ()->DefaultDict(0))
    
    for tree in rf.trees
        row_importance, feature_importance = row_feature_importance(tree, sample)
        for row_index in keys(row_importance)
            row_importance_sums[row_index] += row_importance[row_index]
            for feature_index in keys(feature_importance)
                # (for each row_index, increment the importances of the corresponding features)
                feature_importance_sums[row_index][feature_index] += row_importance[row_index]*feature_importance[feature_index]
            end
        end
    end


    feature_importances = Dict()
    for row_index in keys(feature_importance_sums)
        feature_importances[row_index] = dict_mean(feature_importance_sums[row_index])
    end

    dict_mean(row_importance_sums), feature_importances
end