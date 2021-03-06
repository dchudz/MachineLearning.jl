type BartTreeTransformationProbabilies
    node_birth_death::Float64
    change_decision_rule::Float64
    swap_decision_rule::Float64

    function BartTreeTransformationProbabilies(n, c, s)
        assert(n+c+s==1.0)
        new(n, c, s)
    end
end
BartTreeTransformationProbabilies() = BartTreeTransformationProbabilies(0.5, 0.4, 0.1)

type BartOptions <: RegressionModelOptions
    num_trees::Int
    burn_in::Int
    num_draws::Int
    alpha::Float64
    beta::Float64
    k::Float64
    transform_probabilities::BartTreeTransformationProbabilies
    display::Bool
end
BartOptions() = BartOptions(10, 200, 1000, 0.95, 2.0, BartTreeTransformationProbabilies(), false)

function bart_options(;num_trees::Int=10,
                      burn_in::Int=200,
                      num_draws::Int=1000,
                      alpha::Float64=0.95,
                      beta::Float64=2.0,
                      k::Float64=2.0,
                      transform_probabilities::BartTreeTransformationProbabilies=BartTreeTransformationProbabilies(),
                      display::Bool=false)
    BartOptions(num_trees, burn_in, num_draws, alpha, beta, k, transform_probabilities, display)
end

type Bart <: RegressionModel # This is a trivial holder for the data/options. All the magic happens in predict(...), not fit(...)
    x::Matrix{Float64}
    y_normalized::Vector{Float64}
    y_min::Float64
    y_max::Float64
    options::BartOptions
end

type BartLeafParameters
    sigma::Float64
    sigma_prior::Float64
    nu::Float64
    lambda::Float64
end

type BartLeaf <: DecisionLeaf
    value::Float64
    r_mean::Float64
    r_sigma::Float64
    train_data_indices::Vector{Int}

    function BartLeaf(r::Vector{Float64}, train_data_indices::Vector{Int})
        if length(train_data_indices)==0
            r_mean  = 0.0
            r_sigma = 1.0
        else
            leaf_r  = r[train_data_indices]
            r_mean  = mean(leaf_r)
            r_sigma = sum((leaf_r.-r_mean).^2)
        end

        new(0.0, r_mean, r_sigma, train_data_indices)
    end
end

type BartTree <: AbstractRegressionTree
    tree::DecisionTree
end

type BartState <: RegressionModel
    trees::Vector{BartTree}
    leaf_parameters::BartLeafParameters
end

linear_model_sigma_prior(x::Matrix{Float64}, y::Vector{Float64}) = std(x*(x\y)-y)

function nonterminal_node_prior(alpha::Float64, beta::Float64, depth::Int)
    alpha * depth^(-beta) # root node has depth=1 (note BART paper has depth(root)=0)
end
nonterminal_node_prior(opts::BartOptions, depth::Int) = nonterminal_node_prior(opts.alpha, opts.beta, depth)

function growth_prior(node::DecisionNode, depth::Int, opts::BartOptions)
    indices = train_data_indices(node)
    branch_prior = nonterminal_node_prior(opts, depth)
    if length(indices) >= 5
        return branch_prior
    elseif length(indices) > 0
        return 0.001*branch_prior
    else
        return 0.0
    end
end

function log_node_prior(branch::DecisionBranch, branch_depth::Int, opts::BartOptions)
    indices = train_data_indices(branch)
    prior = log(growth_prior(branch, branch_depth, opts)) - log(length(indices))
    prior + log_node_prior(branch.left, branch_depth+1, opts) + log_node_prior(branch.right, branch_depth+1, opts)
end
log_node_prior(leaf::DecisionLeaf, leaf_depth::Int, opts::BartOptions) = log(1.0 - growth_prior(leaf, leaf_depth, opts))


function log_likelihood(leaf::BartLeaf, params::BartLeafParameters)
    ll = 0.0
    if length(leaf.train_data_indices)==0
        ll = -10000000.0
    else
        a   = 1.0/params.sigma_prior^2.0
        b   = length(leaf.train_data_indices) / params.sigma^2
        ll  = 0.5*log(a/(a+b))
        ll -= leaf.r_sigma^2/(2.0*params.sigma^2)
        ll -= 0.5*a*b*leaf.r_mean^2/(a+b)
    end
    ll
end
log_likelihood(branch::DecisionBranch, params::BartLeafParameters) = log_likelihood(branch.left, params) + log_likelihood(branch.right, params)

function update_sigma!(bart_state::BartState, residuals::Vector{Float64})
    sum_r_sigma_squared = sum(residuals.^2)
    nlpost = bart_state.leaf_parameters.nu*bart_state.leaf_parameters.lambda + sum_r_sigma_squared
    bart_state.leaf_parameters.sigma = sqrt(nlpost / rand(Chisq(bart_state.leaf_parameters.nu + length(residuals))))
end

function update_leaf_values!(tree::BartTree, params::BartLeafParameters)
    for leaf=leaves(tree)
        update_leaf_value!(leaf, params)
    end
end

function update_leaf_value!(leaf::BartLeaf, params::BartLeafParameters)
    a          = 1.0/params.sigma_prior^2.0
    b          = length(leaf.train_data_indices) / params.sigma^2
    post_mu    = b*leaf.r_mean / (a+b)
    post_sigma = 1.0 / sqrt(a+b)
    leaf.value = post_mu + post_sigma*randn()
end

function update_tree!(bart::Bart, bart_state::BartState, tree::BartTree, r::Vector{Float64})
    select_action = rand()
    if select_action < bart.options.transform_probabilities.node_birth_death
        alpha, updated = node_birth_death!(bart, bart_state, tree, r)
    elseif select_action < bart.options.transform_probabilities.node_birth_death + bart.options.transform_probabilities.change_decision_rule
        alpha, updated = change_decision_rule!(bart, bart_state, tree, r)
    else
        alpha, updated = swap_decision_rule!(bart, bart_state, tree, r)
    end
    if updated
        update_leaf_values!(tree, bart_state.leaf_parameters)
    end
    alpha, updated
end

function train_data_indices(branch::DecisionBranch)
    function train_data_indices!(branch::DecisionBranch, indices::Vector{Int})
        train_data_indices!(branch.left,  indices)
        train_data_indices!(branch.right, indices)
    end
    function train_data_indices!(leaf::BartLeaf, indices::Vector{Int})
        for i=leaf.train_data_indices
            push!(indices, i)
        end
    end

    indices = Array(Int, 0)
    train_data_indices!(branch, indices)
    sort(indices)
end
train_data_indices(leaf::DecisionLeaf) = leaf.train_data_indices

function fix_data!(branch::DecisionBranch, x::Matrix{Float64}, r::Vector{Float64}, indices::Vector{Int})
    function fix_data!(parent::DecisionBranch, leaf::BartLeaf, left_child::Bool, x::Matrix{Float64}, r::Vector{Float64}, indices::Vector{Int})
        if left_child
            parent.left  = BartLeaf(r, indices)
        else
            parent.right = BartLeaf(r, indices)
        end
    end
    fix_data!(parent::DecisionBranch, branch::DecisionBranch, left_child::Bool, x::Matrix{Float64}, r::Vector{Float64}, indices::Vector{Int}) = fix_data!(branch, x, r, indices)

    if length(indices)==0
        left_indices  = indices
        right_indices = indices
    else
        feature       = x[indices, branch.feature]
        left_indices  = indices[map(z->z<=branch.value, feature)]
        right_indices = indices[map(z->z >branch.value, feature)]
    end
    fix_data!(branch, branch.left,  true,  x, r, left_indices)
    fix_data!(branch, branch.right, false, x, r, right_indices)
end

function initialize_bart_state(bart::Bart)
    trees = Array(BartTree, 0)
    for i=1:bart.options.num_trees
        push!(trees, BartTree(DecisionTree(BartLeaf(bart.y_normalized, [1:size(bart.x,1)]))))
    end
    sigma  = linear_model_sigma_prior(bart.x, bart.y_normalized)
    nu     = 3.0
    musig  = 0.5/(bart.options.k*sqrt(bart.options.num_trees))
    if bart.options.display
        println("Sigma Hat: ", sigma)
        println("Std Y: ", sqrt(mean(bart.y_normalized.^2)))
        println("MuSig: ", musig)
    end
    q      = 0.90
    lambda = sigma^2.0*quantile(NoncentralChisq(nu, 1.0), q)/nu
    params = BartLeafParameters(sigma, musig, nu, lambda)
    bart_state = BartState(trees, params)
    for tree=bart_state.trees
        update_leaf_values!(tree, bart_state.leaf_parameters)
    end
    bart_state
end

function birth_node(tree::BartTree)
    if typeof(tree.tree.root) == BartLeaf
        leaf = tree.tree.root
        leaf_probability = 1.0
    else
        leaf_nodes = leaves(tree)
        i = rand(1:length(leaf_nodes))
        leaf = leaf_nodes[i]
        leaf_probability = 1.0/length(leaf_nodes)
    end

    leaf, leaf_probability
end
probability_node_birth(tree::BartTree) = typeof(tree.tree.root) == BartLeaf ? 1.0 : 0.5

function node_birth!(bart::Bart, bart_state::BartState, tree::BartTree, r::Vector{Float64}, probability_birth::Float64)
    leaf, leaf_node_probability = birth_node(tree)
    if length(leaf.train_data_indices)==0
        return 0.0, false
    end
    
    leaf_depth    = depth(tree, leaf)
    leaf_prior    = growth_prior(leaf, leaf_depth, bart.options)
    ll_before     = log_likelihood(leaf, bart_state.leaf_parameters)
    split_feature = rand(1:size(bart.x,2))
    split_loc     = rand(1:length(leaf.train_data_indices)) # TODO: throwout invalid splits prior to this
    feature       = bart.x[leaf.train_data_indices, split_feature]
    split_value   = sort(feature)[split_loc]
    left_indices  = leaf.train_data_indices[map(z->z<=split_value, feature)]
    right_indices = leaf.train_data_indices[map(z->z >split_value, feature)]
    left_leaf     = BartLeaf(r, left_indices)
    right_leaf    = BartLeaf(r, right_indices)
    branch        = DecisionBranch(split_feature, split_value, left_leaf, right_leaf)

    left_prior    = growth_prior(left_leaf , leaf_depth+1, bart.options)
    right_prior   = growth_prior(right_leaf, leaf_depth+1, bart.options)
    ll_after      = log_likelihood(branch, bart_state.leaf_parameters)

    parent_branch = parent(tree, leaf)
    num_not_grand_branches = length(not_grand_branches(tree))
    if parent_branch == None 
        num_not_grand_branches += 1
    elseif typeof(parent_branch.left) != BartLeaf || typeof(parent_branch.right) != BartLeaf
        num_not_grand_branches += 1
    end

    p_not_grand = 1.0/num_not_grand_branches
    p_dy  = 0.5 #1.0-probability_node_birth(tree)

    alpha1 = (leaf_prior*(1.0-left_prior)*(1.0-right_prior)*p_dy*p_not_grand)/((1.0-leaf_prior)*probability_birth*leaf_node_probability)
    alpha  = alpha1 * exp(ll_after-ll_before)

    if rand()<alpha
        if parent_branch == None
            tree.tree.root = branch
        else
            if leaf==parent_branch.left
                parent_branch.left  = branch
            else
                parent_branch.right = branch
            end
        end
        updated = true
    else
        updated = false
    end

    alpha, updated
end

function death_node(tree::BartTree)
    not_grand_branch_nodes = not_grand_branches(tree)
    not_grand_branch_nodes[rand(1:length(not_grand_branch_nodes))], 1.0/length(not_grand_branch_nodes)
end

function node_death!(bart::Bart, bart_state::BartState, tree::BartTree, r::Vector{Float64}, probability_death::Float64)
    branch, p_not_grand = death_node(tree)
    leaf_depth          = depth(tree, branch.left)
    left_prior          = growth_prior(branch.left, leaf_depth, bart.options)
    right_prior         = growth_prior(branch.left, leaf_depth, bart.options)
    ll_before           = log_likelihood(branch, bart_state.leaf_parameters)
    leaf                = BartLeaf(r, sort(vcat(branch.left.train_data_indices, branch.right.train_data_indices)))
    ll_after            = log_likelihood(leaf, bart_state.leaf_parameters)

    parent_branch = parent(tree, branch)
    probability_birth_after = parent_branch == None ? 1.0 : 0.5
    prior_grow = growth_prior(leaf, leaf_depth-1, bart.options)
    probability_birth_leaf = 1.0 / (length(leaves(tree))-1)

    alpha1 = ((1.0-prior_grow)*probability_birth_after*probability_birth_leaf)/(prior_grow*(1.0-left_prior)*(1.0-right_prior)*probability_death*p_not_grand)
    alpha  = alpha1*exp(ll_after-ll_before)

    if rand()<alpha
        if parent_branch == None 
            tree.tree.root = leaf
        else
            if parent_branch.left == branch
                parent_branch.left =  leaf
            else
                parent_branch.right = leaf
            end
        end
        updated = true
    else
        updated = false
    end

    alpha, updated
end

function node_birth_death!(bart::Bart, bart_state::BartState, tree::BartTree, r::Vector{Float64})
    probability_birth = probability_node_birth(tree)
    if rand() < probability_birth
        alpha, updated = node_birth!(bart, bart_state, tree, r, probability_birth)
    else
        probability_death = 1.0 - probability_birth
        alpha, updated = node_death!(bart, bart_state, tree, r, probability_death)
    end
    alpha, updated
end

function change_decision_rule!(bart::Bart, bart_state::BartState, tree::BartTree, r::Vector{Float64})
    branch_nodes = branches(tree)
    if length(branch_nodes)==0
        return 0.0, false
    end

    branch       = branch_nodes[rand(1:length(branch_nodes))]
    branch_depth = depth(tree, branch)
    indices      = train_data_indices(branch)

    old_feature  = branch.feature
    old_value    = branch.value
    ll_before    = log_likelihood(branch, bart_state.leaf_parameters)
    prior_before = log_node_prior(branch, branch_depth, bart.options)

    features = [1:size(bart.x,2)]
    splice!(features, branch.feature)
    new_feature = features[rand(1:length(features))]
    new_value   = bart.x[indices[rand(1:length(indices))], new_feature]

    fix_data!(branch, bart.x, r, indices)
    ll_after    = log_likelihood(branch, bart_state.leaf_parameters)
    prior_after = log_node_prior(branch, branch_depth, bart.options)

    alpha = isnan(ll_after + prior_after) ? 0.0 : exp(prior_after + ll_after - prior_before - ll_before)

    if rand()<alpha
        updated = true
    else
        branch.feature = old_feature
        branch.value   = old_value
        fix_data!(branch, bart.x, r, indices)
        updated = false
    end

    alpha, updated
end

function swap_decision_rule!(bart::Bart, bart_state::BartState, tree::BartTree, r::Vector{Float64})
    function swap_decision_rule!(branch::DecisionBranch, child::DecisionBranch, x::Matrix{Float64}, r::Vector{Float64}, indices::Vector{Int})
        feature        = branch.feature
        value          = branch.value
        branch.feature = child.feature
        branch.value   = child.value
        child.feature  = feature
        child.value    = value
        fix_data!(branch, x, r, indices)
    end

    branch_nodes = grand_branches(tree)
    if length(branch_nodes)==0
        return 0.0, false
    end

    branch       = branch_nodes[rand(1:length(branch_nodes))]
    branch_depth = depth(tree, branch)
    indices      = train_data_indices(branch)

    child = (typeof(branch.left) == BartLeaf || (rand() < 0.5 && typeof(branch.right) == DecisionBranch)) ? branch.right : branch.left

    ll_before    = log_likelihood(branch, bart_state.leaf_parameters)
    prior_before = log_node_prior(branch, branch_depth, bart.options)

    swap_decision_rule!(branch, child, bart.x, r, indices)

    ll_after    = log_likelihood(branch, bart_state.leaf_parameters)
    prior_after = log_node_prior(branch, branch_depth, bart.options)

    alpha = isnan(ll_after + prior_after) ? 0.0 : exp(prior_after + ll_after - prior_before - ll_before)

    if rand()<alpha
        updated = true
    else
        swap_decision_rule!(branch, child, bart.x, r, indices)
        updated = false
    end

    alpha, updated
end

normalize(y::Vector{Float64}, y_min, y_max) = (y .- y_min) / (y_max - y_min) .- 0.5
normalize(bart::Bart, y::Vector{Float64})   = normalize(y, bart.y_min, bart.y_max)

function StatsBase.fit(x::Matrix{Float64}, y::Vector{Float64}, opts::BartOptions)
    y_min        = minimum(y)
    y_max        = maximum(y)
    y_normalized = normalize(y, y_min, y_max)
    Bart(x, y_normalized, y_min, y_max, opts)
end

function StatsBase.predict(bart_state::BartState, sample::Vector{Float64})
    sum([predict(tree, sample) for tree=bart_state.trees])
end

function StatsBase.predict(bart::Bart, x_test::Matrix{Float64})
    bart_state = initialize_bart_state(bart)

    y_train_current = predict(bart_state, bart.x)
    y_test_current  = predict(bart_state, x_test)
    y_test_hat      = zeros(size(x_test, 1))
    for i=1:bart.options.num_draws
        updates = 0
        for j=1:bart.options.num_trees
            y_old_tree_train = predict(bart_state.trees[j], bart.x)
            y_old_tree_test  = predict(bart_state.trees[j], x_test)
            residuals = bart.y_normalized - y_train_current + y_old_tree_train
            alpha, updated = update_tree!(bart, bart_state, bart_state.trees[j], residuals)
            updates += updated ? 1 : 0
            y_train_current += predict(bart_state.trees[j], bart.x) - y_old_tree_train
            y_test_current  += predict(bart_state.trees[j], x_test)  - y_old_tree_test
        end
        if i>bart.options.burn_in
            y_test_hat += y_test_current
        end
        update_sigma!(bart_state, y_train_current - bart.y_normalized)
        num_leaves = [length(leaves(tree)) for tree=bart_state.trees]
        if bart.options.display && (log(2, i) % 1 == 0.0 || i == bart.options.num_draws)
            println("i: ", i, "\tSigma: ", bart_state.leaf_parameters.sigma, "\tUpdates:", updates, "\tMaxLeafNodes: ", maximum(num_leaves), "\tMeanLeafNodes: ", mean(num_leaves))
        end
    end
    y_test_hat /= bart.options.num_draws - bart.options.burn_in
    (y_test_hat .+ 0.5) * (bart.y_max - bart.y_min) .+ bart.y_min
end