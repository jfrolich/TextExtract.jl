function chi2(X::DTM, y::Array{Int, 1})
    f_obs = y * X
    chisq = f_obs
    chisq -= f_exp
    chisq .^= 2
    chisq ./= f_exp
    vec(chisq)
end


function ngramize_fancy(tokens::Array{UTF8String,1}, n_grams::Int=2)
    n_tokens = length(tokens)
    for m in 1:n_grams
        for index in 2:(n_tokens - m + 1)
            token = join(tokens[index:(index + m - 1)], " ")
            push!(tokens, token)
        end
    end
    tokens
end

function benchmark()
    time = @elapsed for i in 1:N_REPEAT X = texts_to_dtm(texts, 3) end
    println(time/N_REPEAT)
end
# Fancy ngramize (inspiration Text Analysis package)
# function ngramize_fancy(tokens::Array{UTF8String,1}, n_grams::Int=2)
# 	n_tokens = length(tokens)
#     for m in 1:n_grams
#         for index in 2:(n_tokens - m + 1)
#             token = join(tokens[index:(index + m - 1)], " ")
#             push!(tokens, token)
#         end
#     end
#     tokens
# end
