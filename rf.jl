module Dev
#Pkg.add("RDatasets")
#Pkg.add("DataFrames")
using RDatasets
using DataFrames

iris = dataset("datasets", "iris")
features = array(iris[:, 1:4])
labels   = array(iris[:, :Species])
y = map(x -> x == "setosa" ? 1 : 0, labels)
#writetable("features2.csv", iris)

immutable Leaf
    n::Integer
    sum::Integer
end

immutable Node
    j::Integer
    val::Float32
    left::Union(Leaf,Node)
    right::Union(Leaf,Node)
    prop::Float32
end

obs(leaf::Leaf) = leaf.n
obs(tree::Node) = obs(tree.left) + obs(tree.right)

function describe(n::Node)
    println("OBS: ", obs(n))
    println("PROP: ", n.prop)
    println("Feature ID: ", n.j)
    println("Feature > ", n.val)
    println("YES: # Obs ", obs(n.left))
    println(isa(n.left, Node) ? "NODE" : "LEAF")
    println("NO: # Obs ", obs(n.right))
    println(isa(n.right, Node) ? "NODE" : "LEAF")
end

function describe(l::Leaf)
   println("LEAF")
   println("PROP", l.sum/l.n)
end

function split(X, y, nsf)
    nf = size(X, 2)
    N = length(y)
    right = Int[]
    left  = Int[]
    ig    = -Inf
    th    = -Inf
    f     = 1
    for j in 1:nf
        _right, _left, _th, _ig = bestSplit(X[:,j], y, 0)
        if _ig > ig
            right, left, th, ig, f = _right, _left, _th, _ig, j
        end
    end
    right, left, th, f
end

# Returns threshold of best split of a given variable and IG
# function _bestSplit()
function bestSplit(x, y, nsf::Int)
    order = sortperm(x)
    steps   = x[order]
    y       = y[order]
    max_i     = 1
    max_ig    = 0.0
    count     = 0
    prob      = 0.0
    N         = length(y)
    sumY      = sum(y)
    probTotal = sumY/N
    for i in 1:(length(steps)-1)
        count += y[i]
        probSplit   = (sumY-count)/N
        proportion  = 1-i/N
        println(i,":", y[i], ":",steps[i])

        ig = informationGain(probTotal, probSplit, proportion)
        if ig > max_ig
            println("YES!")
            max_ig   = ig
            max_i    = i
        end
    end
    println("Maxx", max_i, x[max_i])
    order[1:max_i], order[max_i+1:end], steps[max_i], max_ig
end

# Returns entropy before - entropy after
# after splitting at a given threshold
function informationGain(probTotal::Float64, probSplit::Float64, proportion::Float64)
    e = entropy(probTotal)
    entropy(probTotal) - (proportion * entropy(probSplit) + (1-proportion) * entropy((1-probSplit)))
end

# Entropy = - p(a)*log(p(a)) - p(b)*log(p(b))

function entropy(p::Float64)
    if (p == 1 || p == 0)
        return 0
    end
    - p*log(p) - (1-p)*log(1-p)
end


function tree(X,y)
    println("--")
    prop = sum(y)/length(y)
    if !(0 < prop < 1)
        return Leaf(length(y), sum(y))
    end

    right, left, th, j = split(X,y,0)
    println("-")
    X_left  = X[left,:]    
    X_right = X[right,:]
    y_left  = y[left]
    y_right = y[right]
    println("==")
    println(length(y))
    println(length(y_left))
    println(length(y_right))
    if !(0 < sum(y_left)/length(y_left) < 1)
        return Node(j, th, Leaf(length(y_left), sum(y_left)), tree(X_right, y_right), prop)
    elseif !(0 < sum(y_right)/length(y_right) < 1)
        return Node(j, th, tree(X_left, y_left), Leaf(length(y_right), sum(y_right[1])), prop)
    else
        return Node(j, th, tree(X_left, y_left), tree(X_right, y_right), prop)
    end
end

function walk(n::Node, x)
    return x[n.j] > n.val ? walk(n.left, x) : walk(n.right, x)
end

function walk(l::Leaf, x)
    return l.maj
end

function predict(n::Node, X)
    y = Float64[]
    for i in 1:size(X)[1]
       push!(y, walk(n, X[i,:]))
    end
    y
end
# Dev module

dt = Dev.tree(Dev.features, Dev.y)
end
