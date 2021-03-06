abstract NeuralNetStopCriteria

type StopAfterIteration <: NeuralNetStopCriteria
    max_iteration::Int
end
StopAfterIteration() = StopAfterIteration(100)

type StopAfterValidationErrorStopsImproving <: NeuralNetStopCriteria
    validation_set_size::Float64
    validation_iteration_window_size::Int
    min_iteration::Int
    max_iteration::Int
end
StopAfterValidationErrorStopsImproving() = StopAfterValidationErrorStopsImproving(0.2, 2, 10, 1000)

type NeuralNetOptions <: ClassificationModelOptions
    bias_unit::Bool # include a bias unit that always outputs a +1
    hidden_layers::Vector{Int} # sizes of hidden layers
    train_method::Symbol
    learning_rate::Float64
    stop_criteria::NeuralNetStopCriteria
    display::Bool
    track_cost::Bool
end

function neural_net_options(;bias_unit::Bool=true,
                            hidden_layers::Vector{Int}=[50],
                            train_method::Symbol=:sgd,
                            learning_rate::Float64=10.0,
                            stop_criteria::NeuralNetStopCriteria=StopAfterIteration(),
                            display::Bool=false,
                            track_cost=false)
    NeuralNetOptions(bias_unit, hidden_layers, train_method, learning_rate, stop_criteria, display, track_cost)
end

type NeuralNetLayer
    weights::Matrix{Float64}
end

type NeuralNet <: ClassificationModel
    options::NeuralNetOptions
    layers::Vector{NeuralNetLayer}
    classes::Vector
end

type NeuralNetTemporary # prevents us from making unnecessary allocations
    outputs::Vector{Matrix{Float64}}
    activations::Vector{Matrix{Float64}}
    deltas::Vector{Matrix{Float64}}
    layer_gradients::Vector{Matrix{Float64}}
end

function initialize_neural_net_temporary(net::NeuralNet)
    outputs = Array(Matrix{Float64}, 0)
    activations = Array(Matrix{Float64}, 0)
    deltas = Array(Matrix{Float64}, 0)
    layer_gradients = Array(Matrix{Float64}, 0)
    num_features = size(net.layers[1].weights, 2) - (net.options.bias_unit ? 1 : 0)
    num_features_with_bias = size(net.layers[1].weights, 2)

    push!(outputs, Array(Float64, (num_features, 1)))
    push!(activations, Array(Float64, (num_features_with_bias, 1)))
    push!(deltas, Array(Float64, (num_features, 1))) # note: deltas[1] never used
    for (i, layer) = enumerate(net.layers)
        num_nodes = size(layer.weights, 1)
        push!(outputs, Array(Float64, (num_nodes, 1)))
        push!(activations, Array(Float64, (num_nodes + (net.options.bias_unit && i<length(net.layers) ? 1 : 0), 1)))
        push!(deltas, Array(Float64, (num_nodes, 1)))
        push!(layer_gradients, similar(layer.weights))
    end

    NeuralNetTemporary(outputs, activations, deltas, layer_gradients)
end

function classes(net::NeuralNet)
    net.classes
end

function sigmoid(z::Array{Float64})
    @devec res = 1./(1+exp(-z))
end

function sigmoid_gradient(z::Array{Float64})
    sz = sigmoid(z)
    @devec res = sz .* (1-sz)
end

function one_hot(y::Vector, classes_map::Dict)
    values = zeros(length(y), length(classes_map))
    for i=1:length(y)
        values[i, classes_map[y[i]]] = 1.0
    end
    values
end

function StatsBase.fit(x::Matrix{Float64}, y::Vector, opts::NeuralNetOptions)
    num_features = size(x, 2)
    classes = sort(unique(y))
    classes_map = Dict(classes, [1:length(classes)])
    num_classes = length(classes)
    net = initialize_net(opts, classes, num_features)
    temp = initialize_neural_net_temporary(net)

    if opts.train_method==:sgd # stochastic gradient descent
        if typeof(opts.stop_criteria)==StopAfterIteration
            train_preset_stop!(net, x, one_hot(y, classes_map), temp)
        elseif typeof(opts.stop_criteria)==StopAfterValidationErrorStopsImproving
            split = split_train_test(x, y, split_fraction=opts.stop_criteria.validation_set_size)
            x_train, y_train = train_set(split)
            x_val,   y_val   = test_set(split)
            train_valid_stop!(net, x_train, one_hot(y_train, classes_map), x_val, one_hot(y_val, classes_map), temp)
        end
    else
        # use optimize from Optim.jl
        actuals = one_hot(y, classes_map)
        initial_weights = net_to_weights(net)

        f = weights -> cost(net, x, actuals, weights)
        g! = (weights, gradients) -> cost_gradient_update_net!(net, x, actuals, weights, gradients, temp)
        res = optimize(f, g!, initial_weights, method=opts.train_method)
        weights_to_net!(res.minimum, net)
        net
    end
    net
end

function train_preset_stop!(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64}, temp::NeuralNetTemporary)
    num_samples = size(x,1)
    for iter=1:net.options.stop_criteria.max_iteration
        if net.options.display
            println("Iteration ", iter)
        end
        if net.options.track_cost
            println("  Cost: ", cost(net, x, actuals))
        end
        for j=1:num_samples
            update_weights!(net, vec(x[j,:]), vec(actuals[j,:]), net.options.learning_rate, num_samples, temp)
        end
    end
end

function train_valid_stop!(net::NeuralNet,
                           x_train::Matrix{Float64},
                           a_train::Matrix{Float64},
                           x_val::Matrix{Float64},
                           a_val::Matrix{Float64},
                           temp::NeuralNetTemporary)
    num_samples = size(x_train,1)
    validation_scores = Array(Float64, 0)
    
    iteration = 0
    while iteration<net.options.stop_criteria.max_iteration
        iteration += 1
        if net.options.display
            println("Iteration ", iteration)
        end
        if net.options.track_cost
            println("  Training Cost: ", cost(net, x_train, a_train))
        end
        for j=1:num_samples
            update_weights!(net, vec(x_train[j,:]), vec(a_train[j,:]), net.options.learning_rate, num_samples, temp)
        end
        preds = predict_probs(net, x_val)
        err = mean_log_loss(a_val, preds)
        push!(validation_scores, err)
        if iteration>=2*net.options.stop_criteria.validation_iteration_window_size && iteration>net.options.stop_criteria.min_iteration
            ws = net.options.stop_criteria.validation_iteration_window_size
            if minimum(validation_scores[iteration-ws+1:iteration])>=maximum(validation_scores[iteration-2*ws+1:iteration-ws])
                break
            end
        end
    end
    println("Number of Iterations: ", iteration)
end

function predict_probs(net::NeuralNet, sample::Vector{Float64})
    state = sample
    for layer = net.layers
        if net.options.bias_unit==true
            state = [1.0;state]
        end

        state = sigmoid(layer.weights*state)
    end
    state
end

function StatsBase.predict(net::NeuralNet, sample::Vector{Float64})
    probs = predict_probs(net, sample)
    net.classes[minimum(find(x->x==maximum(probs), probs))]
end

function update_weights!(net::NeuralNet, sample::Vector{Float64}, actual::Vector{Float64}, learning_rate::Float64, num_samples::Int, temp::NeuralNetTemporary)
    cost_gradient!(net, sample, actual, temp)
    regularization_gradient!(net, temp, 1.0/num_samples)

    for i=1:length(net.layers)
        net.layers[i].weights -= learning_rate*temp.layer_gradients[i]/num_samples
    end
end

function initialize_layer(number_in::Int, number_out::Int)
    epsilon_init = sqrt(6) / sqrt(number_in + number_out)
    weights = 2.0 * (rand(number_out, number_in) .- 0.5) * epsilon_init
    NeuralNetLayer(weights)
end

function initialize_net(opts::NeuralNetOptions, classes::Vector, num_features::Int)
    layers = Array(NeuralNetLayer, 0)
    if isempty(opts.hidden_layers)
        push!(layers, initialize_layer(num_features + (opts.bias_unit?1:0), length(classes)))
    else
        push!(layers, initialize_layer(num_features + (opts.bias_unit?1:0), opts.hidden_layers[1]))
        for i=1:length(opts.hidden_layers)-1
            push!(layers, initialize_layer(opts.hidden_layers[i] + (opts.bias_unit?1:0), opts.hidden_layers[i+1]))
        end
        push!(layers, initialize_layer(opts.hidden_layers[length(opts.hidden_layers)] + (opts.bias_unit?1:0), length(classes)))
    end
    NeuralNet(opts, layers, classes)
end

function weights_to_net!(weights::Vector{Float64}, net::NeuralNet)
    loc = 0
    for layer = net.layers
        layer.weights[:] = weights[loc+1:loc+length(layer.weights)]
        loc += length(layer.weights)
    end
    @assert loc==length(weights)
end

function net_to_weights(net::NeuralNet)
    weights = Array(Float64, 0)
    for layer=net.layers
        for w=layer.weights
            push!(weights, w)
        end
    end
    weights
end

function cost(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64})
    @assert size(x,1)==size(actuals,1)
    probs = predict_probs(net, x)
    err = mean_log_loss(actuals, probs)
    regularization = 0.0
    for layer=net.layers
        w = copy(layer.weights)
        if net.options.bias_unit
            w[:,1] = 0.0
        end
        regularization += sum(w.^2)/(2*size(x,1))
    end
    err + regularization
end

function cost(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64}, weights::Vector{Float64})
    weights_to_net!(weights, net)
    cost(net, x, actuals)
end

function copy_range!(destination::Array, destination_start::Int, source::Array, source_start::Int)
    for i=source_start:length(source)
        destination[destination_start+i-source_start] = source[i]
    end
end

function cost_gradient!(net::NeuralNet, sample::Vector{Float64}, actual::Vector{Float64}, temp::NeuralNetTemporary)
    num_layers = length(net.layers)
    copy!(temp.outputs[1], sample)
    if net.options.bias_unit
        temp.activations[1][1] = 1.0
        copy_range!(temp.activations[1], 2, sample, 1)
    else
        copy!(temp.activations[1], sample)
    end

    for i=1:num_layers
        A_mul_B!(temp.outputs[i+1], net.layers[i].weights, temp.activations[i])
        if net.options.bias_unit && i<num_layers
            temp.activations[i+1][1] = 1.0
            copy_range!(temp.activations[i+1], 2, sigmoid(temp.outputs[i+1]), 1)
        else
            copy!(temp.activations[i+1], sigmoid(temp.outputs[i+1]))
        end
    end

    copy!(temp.deltas[num_layers+1], temp.activations[num_layers+1] - actual)
    for i=num_layers:-1:1
        A_mul_B!(temp.layer_gradients[i], temp.deltas[i+1], temp.activations[i]')
        if i>1
            if net.options.bias_unit
                copy_range!(temp.deltas[i], 1, net.layers[i].weights'*temp.deltas[i+1], 2)
            else
                copy!(temp.deltas[i], net.layers[i].weights'*temp.deltas[i+1])
            end
            copy!(temp.deltas[i], temp.deltas[i].*sigmoid_gradient(temp.outputs[i]))
        end
    end
end

function regularization_gradient(net::NeuralNet)
    layer_gradients = [copy(layer.weights) for layer=net.layers]
    for i=1:length(layer_gradients)
        if net.options.bias_unit
            layer_gradients[i][:,1]=0.0
        end
    end
    layer_gradients
end

function regularization_gradient!(net::NeuralNet, temp::NeuralNetTemporary, lambda::Float64)
    for i=1:length(net.layers)
        start_col = net.options.bias_unit ? 2 : 1
        temp.layer_gradients[i][:,start_col:end] += lambda*net.layers[i].weights[:,start_col:end]
    end
end

function cost_gradient_update_net!(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64}, weights::Vector{Float64}, gradients::Vector{Float64}, temp::NeuralNetTemporary)
    @assert size(x,1)==size(actuals,1)
    weights_to_net!(weights, net)
    gradients[:]=0.0
    layer_gradients = [0.0*layer.weights for layer=net.layers]
    for i=1:size(x,1)
        cost_gradient!(net, vec(x[i,:]), vec(actuals[i,:]), temp)
        layer_gradients += temp.layer_gradients/size(x,1)
    end
    regularization = regularization_gradient(net)
    layer_gradients += regularization/size(x,1)

    i=0
    for gradient=layer_gradients
        for g=gradient
            i += 1
            gradients[i] = g
        end
    end
end

function Base.show(io::IO, net::NeuralNet)
    info = join(["Neural Network",
                 @sprintf("    %d Hidden Layers",length(net.options.hidden_layers)),
                 @sprintf("    %d Classes",length(net.classes))], "\n")
    print(io, info)
end