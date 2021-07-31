using Tables
using MLJ
using CSV: File
using BSON: @save, @load
using DataFrames
using Plots
using LatinHypercubeSampling

load_lcs(file::String) = DataFrame(File(file))
d_smc = load_lcs("data/carac_smc")	
d_lmc = load_lcs("data/carac_lmc")

DataFrames.transform!(d_smc, :obs => (v -> v .== "CC") => :iceph)
DataFrames.transform!(d_lmc, :obs => (v -> v .== "CC") => :iceph)


feature_keys = [:med, :mad, :os, :low, :row, :rAbbe, 
                :abs_dis, :dis, :abs_lin_res, :lin_res,
                :amplitude, :per1, :per2,:per3, :per4]

inside(feature:: Symbol) = feature in feature_keys
y_test, X_test =  unpack(d_smc,
               ==(:iceph),            # y is the :Exit column
               colname -> inside(colname);            # X is the rest, except :Time
			    #wrap_singles=true,
               :med=>MLJ.Continuous,
               :iceph=>OrderedFactor)

y_train, X_train =  unpack(d_lmc,
              ==(:iceph),            # y is the :Exit column
              colname -> inside(colname);            # X is the rest, except :Time
              #wrap_singles=true,
			  :med=>MLJ.Continuous,
              :iceph=>OrderedFactor)	

Model = MLJ.@load RandomForestClassifier pkg=DecisionTree add=true
model = Model()

r = [range(model, :max_depth, lower=2, upper=14.0, scale=:linear),
	 range(model, :n_subfeatures, lower=2, upper=14.0, scale=:linear),
	 range(model, :n_trees, lower=5, upper=100, scale=:log)]
self_tuning_tree = TunedModel(model=model,
                              resampling=CV(nfolds=5),
                              tuning=Grid(resolution=8),
                              range=r,
							  operation=predict_mode,
                              measure=balanced_accuracy);
mach = machine(self_tuning_tree, X_test, y_test);
println("Fitting");
fit!(mach)
	
measure_mach = machine( fitted_params(mach).best_model,	X_test,y_test)
ŷ = predict_mode(mach, X_train)
balanced_accuracy(y_train,ŷ), MLJ.evaluate!(measure_mach, resampling=StratifiedCV(nfolds=6, shuffle=true),
    measure=confmat, operation=predict_mode)

