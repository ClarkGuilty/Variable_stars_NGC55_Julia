using Tables
using MLJ

using CSV: File
using BSON: @save, @load
using DataFrames

load_lcs(file::String) = DataFrame(File(file))

d_lmc = load_lcs("data/carac_lmc")
d_smc = load_lcs("data/carac_smc")

feature_keys = [:med, :mad, :os, :low, :row, :rAbbe, 
                :abs_dis, :dis, :abs_lin_res, :lin_res,
                :amplitude, :per1, :per2,:per3, :per4]

#test_features = d_smc[:, feature_keys]
#train_features = d_lmc[:, feature_keys]

#test_target = d_smc[:, [end]]
#test_target = d_lmc[:, [end]]


inside(cosa:: Symbol) = cosa in feature_keys
y_test, X_test =  unpack(d_smc,
               ==(:obs),            # y is the :Exit column
               colname -> inside(colname);            # X is the rest, except :Time
               :med=>Continuous,
               :obs=>Multiclass)

y_train, X_train =  unpack(d_lmc,
              ==(:obs),            # y is the :Exit column
              colname -> inside(colname);            # X is the rest, except :Time
              :med=>Continuous,
              :obs=>Multiclass)

#Xfixed = coerce(X, AbstractFloat=>Continuous)
#X_2  = scitype.(Xfixed)
#Tree = @load DecisionTreeClassifier pkg=DecisionTree
#models(matching(X_2,scitype.(y)))

models(matching(X_train,y_train))


models() do model
    matching(model, X_train, y_train) &&
    model.is_pure_julia
end


#Tree = @iload RandomForestClassifier pkg=DecisionTree
#tree = Tree()

#Tree = @iload ConstantClassifier pkg=MLJModels
#tree = Tree()

#Tree = @iload DecisionTreeClassifier pkg=BetaML
#tree = Tree()

Tree = @iload RandomForestClassifier pkg=BetaML
tree = Tree(maxDepth = 2)

mach = machine(tree, X_test, y_test)

fit!(mach)

mean(cross_entropy(yhat, y_train))
#evaluate(mach, resampling=CV(nfolds=5), measure=[RootMeanSquaredError(), MeanAbsoluteError()])


#predict_mode(mach, X_test) # a vector of point-predictions
