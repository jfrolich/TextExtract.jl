library('tm')
library('RWeka')

timing = c('R', toString(Sys.time()))

elapsed <- function(f) {  
  time = (system.time({
    (function() f)()
  }))
  time[3]
}

PATH = "C:\\Users\\JFrolich\\Documents\\Data sets\\Reuters\\fetch\\full_dataset"
metadata = read.table(sprintf("%s\\%s", PATH, 'metadata.csv'), sep=",", header=TRUE)
y = metadata$earn

timing = append(timing, elapsed({
  texts = apply(array(metadata$id), 1, function(id) {
    paste(readLines(sprintf('%s\\text\\%s', PATH, id)), collapse = ' ')
  })
  corpus <-  Corpus(VectorSource(texts, encoding='utf-8'))
}))

timing = append(timing, elapsed({
  TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 3))
  DocumentTermMatrix(corpus, control = list(tokenize = TrigramTokenizer))
}))

print(paste(timing, collapse = ', '))
