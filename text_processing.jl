module TextProcessing

# The vocabulary is a dictionary with all the terms as keys
# and the column in the data matrix as values
typealias Vocabulary Dict{UTF8String, Uint32}
typealias UnweightedDTM SparseMatrixCSC{Uint32}
typealias WeightedDTM SparseMatrixCSC{Float64}
typealias DTM Union(WeightedDTM, UnweightedDTM)
typealias Document UTF8String

export preprocess,
			 tokenize,
			 text_files,
			 text,
			 text_array,
			 chi2,
			 select_k_best,
			 ngramnize,
			 text_from_files,
			 dtm,
			 vocabulary_to_feature_names

import Base.length

function preprocess(d::Document)
	lowercase(d)
end

function tokenize(s::Document)
	[utf8(s) for s in matchall(r"\b\w\w+\b", preprocess(s))]
end

# A document collection is a collection of documents.
# The implementation needs the following interface:
#
# text: returns the plain text for the given document
# length: returns the length of the collection
#
abstract DocumentCollection

# Text files implementation
# This implementation of documentcollection iterate through
# text files to create the DTM
type TextFiles <: DocumentCollection
	filenames::Vector{String}
	buffer_ptr::Int
	buffer_size::Int
	buffer::Vector{Document}
end

TEXT_BUFFER_SIZE = 30000

function text_files(files)
	TextFiles(files, -1, TEXT_BUFFER_SIZE, [])
end

# returns the text for index 'i', it pre-filles a buffer of text
# for all the files in the collection when necesary, because
# loading it file by file is much slower (factor 3)
function text(dc::TextFiles, i::Int)
	if dc.buffer_ptr < 0 || i >= (dc.buffer_ptr + dc.buffer_size) || i < dc.buffer_ptr
		fill_buffer!(dc, i)
	end

	dc.buffer[i-dc.buffer_ptr+1]
end

function fill_buffer!(dc::TextFiles, i::Int)
	dc.buffer_ptr = i
	dc.buffer = Document[]
	for i in dc.buffer_ptr:(dc.buffer_ptr + dc.buffer_size)
		if i > length(dc)
			break
		end
		push!(dc.buffer, open(dc.filenames[i]) do f
			Document(readbytes(f))
		end)
	end
end

length(dc::TextFiles) = Base.length(dc.filenames)

# TextArray: This implementation of documentcollection is simply
# given as an array of strings
type TextArray <: DocumentCollection
	texts::Vector{Document}
end

function text(dc::TextArray, i::Int)
	dc.texts[i]
end

function text_array(texts)
	TextArray(texts)
end

length(dc::TextArray) = Base.length(dc.texts)

Base.start(dc::DocumentCollection) = 1
Base.next(dc::DocumentCollection, i) = (text(dc, i), i+1)
Base.done(dc::DocumentCollection, i) = i > length(dc)

# calculates Chi squared scores of a sparse matrix on a binary variable (y)
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

# Feature selection
function select_k_best(X::DTM, y::Vector{Int}, score_fnc, k::Int)
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
	texts = Document[]
	for file in files
		push!(texts, open(file) do f
			UTF8String(readbytes(f))
		end)
	end
	texts
end

# TODO: Add ngram range feature
function dtm(dc::DocumentCollection, n_grams::Int)
	vocabulary = Vocabulary()
	is = Uint32[]
	js = Uint32[]
	vs = Uint32[]
	for (i, text) in enumerate(dc)
		tokens::Array{UTF8String,1} = ngramize(tokenize(text), n_grams)
		for token in tokens
			push!(is, i)
			push!(js, get!(vocabulary, token, length(vocabulary)+1))
			push!(vs, 1)
		end
	end
	(sparse(is, js, vs, length(dc), maximum(js)),
	 vocabulary_to_feature_names(vocabulary))
end

function vocabulary_to_feature_names(vocabulary::Vocabulary)
	inv_voc = Dict{Uint32, UTF8String}()
	for (k,v) in vocabulary
		inv_voc[v] = k
	end
	[inv_voc[convert(Uint32, i)] for i in 1:length(inv_voc)]
end

end # Module Textprocessing
