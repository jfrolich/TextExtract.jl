using DataFrames

PATH = "C:\\Users\\JFrolich\\Documents\\Data sets\\Reuters\\fetch\\full_dataset"
metadata = readtable(string(PATH, "/", "metadata.csv"))
y = array(metadata[:acq])

typealias Vocabulary Dict{UTF8String, Uint32}
typealias UnweightedDTM SparseMatrixCSC{Uint32}
typealias WeightedDTM SparseMatrixCSC{Float64}
typealias DTM Union(WeightedDTM, UnweightedDTM)

function tokenize(s)
	[utf8(s) for s in matchall(r"\b\w\w+\b", lowercase(utf8(s)))]
end

type DocumentCollection
end

type TextFileDocumentCollection <: DocumentCollection
	filenames::Vector{String}
end

function text(f::TextFileDocumentCollection, i)
	open(file) do f
		UTF8String(readbytes(f))
	end
end

length(TextFileDocumentCollection) = length(filenames)
next(dc::DocumentCollection, i) = text(dc, i)
done(dc::DocumentCollection, i) = i > length(dc)

# Feature selection
function chi2(X::DTM, y::Array{Int, 1})
    f_obs = y * X
	feature_count = ones(Float64, size(X)[1]) * X
	class_prob    = mean(y)
    f_exp = (class_prob * feature_count)'
    chisq = Float64[]
	for i in 1:length(f_obs)
		push!(chisq, (f_obs[i] - f_exp[i])^2/(f_exp[i]))
	end
   chisq
end

function select_k_best(X, y, score_fnc, k)
	scores = score_fnc(X,y)
	sortperm(scores::Array{Float64,1}, rev=true, alg=MergeSort)[1:k]
end

function ngramize(tokens::Array{UTF8String,1}, n_grams::Int64=2)
	n_tokens::Int64 = length(tokens)
	for i in 1:n_tokens
		gram = ""
		for j in 1:n_grams
			token = tokens[i+j-1]
			if j == 1
				gram = token
			elseif j > 1 && i+j-1 <= n_tokens
				gram = string(gram, " ", token)
				push!(tokens, gram)
		    end
		end
	end
	tokens
end

function text_from_files(files)
	texts = UTF8String[]
	for file in files
		push!(texts, open(file) do f
			UTF8String(readbytes(f))
		end)
	end
	texts
end

# TODO: create text iterable
# that can iterate through text and text files
# + ngram range

function texts_to_dtm(texts::Array{UTF8String,1}, n_grams)
	vocabulary = Vocabulary()
	is = Uint32[]
	js = Uint32[]
	vs = Uint32[]

	for (i, text) in enumerate(texts)
		tokens::Array{UTF8String,1} = ngramize(tokenize(text), n_grams)

		for token in tokens
			push!(is, i)
			push!(js, get!(vocabulary, token, length(vocabulary)+1))
			push!(vs, 1)
		end
	end
	(sparse(is, js, vs, length(texts), maximum(js)), vocabulary_to_feature_names(vocabulary))
end

function vocabulary_to_feature_names(vocabulary::Vocabulary)
	inv_voc = Dict{Uint32, UTF8String}()
	for (k,v) in vocabulary
		inv_voc[v] = k
	end
	[inv_voc[convert(Uint32, i)] for i in 1:length(inv_voc)]
end

files = [string(PATH, "/text/", id) for id in (metadata[:id])]

dc = TextFileDocumentCollection(files)

timing_1 = @elapsed texts = text_from_files(files)
timing_2 = @elapsed X, feature_names = texts_to_dtm(texts,3)
timing_3 = @elapsed sel_features = select_k_best(X, y, chi2, 1000)

println('Julia,', now(), ',', timing_1, ',', timing_2, ',', timing_3 )
