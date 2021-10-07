library(keras)
source("0-ready_fasta.R")
acceptor <- ready_fasta("../data-in/acceptor.183.fasta")
donor <- ready_fasta("../data-in/donor.183.fasta")
intron <- ready_fasta("../data-in/intron.183.random.fasta")

# Load in FASTA files
x <- c(acceptor, donor, intron)
y <-
  c(rep("acceptor", length(acceptor)),
    rep("donor", length(donor)),
    rep("intron", length(intron))
  )

# Remove sequence with N
anyN <- sapply(x, function(seq) grepl("N", seq))
table(anyN) # 28 TRUE
x <- x[!anyN]
y <- y[!anyN]

# Convert into 1-hot encoding
x_1hot <- ready_dna_bulk(x)
dim(x_1hot)
length(y)

# Split in training/test
set.seed(1)
index <- sample(1:length(x), size = .67*length(x), replace = FALSE)
train.x <- x_1hot[index,,]
train.y <- y[index]
test.x <- x_1hot[-index,,]
test.y <- y[-index]

# Save as RData
system("mkdir ../data-out")
save(train.x, file = "../data-out/p183-train-x.RData")
save(train.y, file = "../data-out/p183-train-y.RData")
save(test.x, file = "../data-out/p183-test-x.RData")
save(test.y, file = "../data-out/p183-test-y.RData")
