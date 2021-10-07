library(keras)
library(caress)
results_dir <- "../results/H-RFE-30"
system(paste0("mkdir ", results_dir))
shannon <- function(p){
  p <- p[p>0]
  -1 * sum(p*log(p))
}

# Load in training and test data
files <- list.files("../data-out/", pattern = "183", full.names = TRUE)
for(file in files){
  load(file)
}

# Make categorical for 3-class problem
train.y_cat <- to_categorical(as.numeric(factor(train.y))-1)
test.y_cat <- to_categorical(as.numeric(factor(test.y))-1)

# Load in AM PWMs
files <- list.files("../results/A-ascend/", pattern = "-pwm.csv",
                    recursive = TRUE, full.names = TRUE)

step <- 30
lambda <- 0
M <- 103

####################################################
### Process data
####################################################

# Load in the PWMs learned by gradient ascent
files_pwm <- files[grepl(paste0("step-", step), files)]
pwms <- lapply(files_pwm, read.csv, row.names = 1)

# Convert PWMs into CNN filter
W_catch <- array(0, c(103, 4, 1, 160))
for(pwm in 1:length(pwms)){
  
  # Scale PWM weights by entropy
  pwm_i <- pwms[[pwm]]
  ic_content <- apply(pwm_i, 2, function(m){
    (shannon(c(.25, .25, .25, .25)) - shannon(m)) /
      shannon(c(.25, .25, .25, .25))
  })
  new_weights <- sweep(pwm_i, 2, ic_content, "*")
  W_catch[,,1,pwm] <- t(new_weights)
}

# Make architecture to get max AM PWMs
k_clear_session()
use_session_with_seed(1)
start <- from_input(train.x)
conv1 <- start %>%
  layer_reshape(c(dim(train.x)[-1], 1)) %>%
  layer_conv_2d(filters = 32*5, kernel_size = c(103, 4), bias_initializer = "zeros") %>%
  layer_global_max_pooling_2d()
m_maxes <- prepare(start, conv1)

# Replace weights with AM PWMs
W_init <- get_layer_weights(m_maxes, "conv2d")
W_init[[1]] <- W_catch
set_layer_weights(m_maxes, "conv2d", weights = W_init, freeze = TRUE)

# Compute max AM PWMs & normalize
train.x_maxes <- predict(m_maxes, train.x)
train.x_maxes_swept <- sweep(train.x_maxes, 2, colMeans(train.x_maxes), "-")
train_sd <- apply(train.x_maxes, 2, sd)
train.x_maxes_swept <- sweep(train.x_maxes_swept, 2, train_sd, "/")

# Compute max AM PWMs & normalize
test.x_maxes <- predict(m_maxes, test.x)
test.x_maxes_swept <- sweep(test.x_maxes, 2, colMeans(train.x_maxes), "-")
test.x_maxes_swept <- sweep(test.x_maxes_swept, 2, train_sd, "/")

# Subset to include only top 30 from RFE
rfe_import <- read.csv("../data-out/30-motifs-after-RFE.csv")
col_keep <- as.numeric(gsub("V", "", colnames(rfe_import)))
train.x <- train.x_maxes_swept[,col_keep]
test.x <- test.x_maxes_swept[,col_keep]

####################################################
###
####################################################

# Now use the maxes...
k_clear_session()
use_session_with_seed(1)
start <- from_input(train.x)
target <- start %>%
  layer_dense(3, activation = "softmax", kernel_regularizer = regularizer_l1(l = lambda))
m <- prepare(start, target)

# Run model
history <- build(m, train.x, train.y_cat, batch_size = 1024, epochs = 250,
                 early_stopping_patience = 250)

png(paste0(results_dir, "/history-step-", step, "-lambda-", lambda, ".png"))
plot(history)
dev.off()

####################################################################
### SAVE RESULTS
####################################################################

png(paste0(results_dir, "/history-step-", step, "-lambda-", lambda, ".png"))
plot(history)
dev.off()

# Get TRAIN SET ACC <- replace with TEST SET ACC later!
yhat <- predict(m, test.x)
yhat_cat <- apply(yhat, 1, which.max)
y_cat <- apply(test.y_cat, 1, which.max)
conf <- table(yhat_cat, y_cat)
write.csv(conf, file = paste0(results_dir, "/conf-step-", step, "-lambda-", lambda, ".csv"))

# 3-class acc
acc <- sum(diag(conf)) / sum(conf)
df <- data.frame(method = "student-ic_scale", step = step, acc = acc, lambda = lambda, M = M, type = "3class")
write.csv(df, file = paste0(results_dir, "/acc-step-", step, "-lambda-", lambda, "-3class.csv"))

# 2-class acc
acc <- (conf[1,1] + conf[2,2]) / (conf[1,2] + conf[2,1] + conf[1,1] + conf[2,2])
df <- data.frame(method = "student-ic_scale", step = step, acc = acc, lambda = lambda, M = M, type = "2class")
write.csv(df, file = paste0(results_dir, "/acc-step-", step, "-lambda-", lambda, "-2class.csv"))

# Get weights assigned to AM PWMs
class_weights <- get_layer_weights(m, "dense")[[1]]
rownames(class_weights) <- files_pwm[col_keep]
colnames(class_weights) <- levels(factor(train.y))
write.csv(class_weights, file = paste0(results_dir, "/class_weights-step-", step, "-lambda-", lambda, ".csv"))
