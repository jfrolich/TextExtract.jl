include("text_processing.jl")

# Benchmark Julia TextProcessing Library

using DataFrames
using Datetime
using TextProcessing

PATH = "C:\\Users\\JFrolich\\Documents\\Data sets\\Reuters\\fetch\\full_dataset"
metadata = readtable(string(PATH, "/", "metadata.csv"))
y = array(metadata[:acq])
files = [string(PATH, "/text/", id) for id in (metadata[:id])]

#dc = text_files(files)
dc = text_array(text_from_files(files))

@profile X, feature_names = dtm(dc,3)
Profile.print()

#println(@elapsed X, feature_names = dtm(dc,3))

#timing_1 = @elapsed text_array(text_from_files(files))
#timing_2 = @elapsed X, feature_names = dtm(dc,3)
#timing_3 = @elapsed sel_features = select_k_best(X, y, chi2, 10000)

#println("Julia,", now(), ",", timing_1, ",", timing_2, ",", timing_3)
