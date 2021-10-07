library(keras)
library(caress)
results_dir <- "../results/J-RFE-30-DISTR"
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
files <- list.files("../results/A-ascend/", pattern = "-step-30-pwm.csv",
                    recursive = TRUE, full.names = TRUE)

####################################################
### Process data
####################################################

# Load in the PWMs learned by gradient ascent
pwms <- lapply(files, read.csv, row.names = 1)

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

####################################################
### Deploy convolutional layers
####################################################

# Load in RFE values
rfe_import <- read.csv("../data-out/30-motifs-after-RFE.csv")
col_keep <- as.numeric(gsub("V", "", colnames(rfe_import)))

convs <- get_layer_output(m_maxes, test.x, "conv2d")
maxes <- predict(m_maxes, test.x)
res <- vector("list", 160)
for(filter in col_keep){
  
  file_i <- paste0(basename(dirname(files[filter])), "-",
                   gsub(".csv", "", basename(files[filter]))) # get name of filter
  filter_over_space <- convs[,,,filter] # get filter scores per position per motif
  max_locs <- apply(filter_over_space, 1, which.max) # get location of max scores per motif
  max_vals <- maxes[,filter] # get value of max scores per motif
  
  library(ggplot2)
  df <- data.frame(file_i, max_locs, max_vals, test.y)
  g <- ggplot(df, aes(x = max_locs, col = test.y)) + geom_histogram(fill = "white", position = "identity") +
    theme_bw()+ labs(col = "Sequence") + ylab("Frequency of Max Score Occurrence\n(Test Data Only)") +
    xlab("Location of Max Score in Aligned Sequence")
  
  png(paste0(results_dir, "/", file_i, ".png"), width = 4, height = 4, res = 600, units = "in")
  plot(g)
  dev.off()
  
  res[filter] <- list(df)
}
res <- res[!sapply(res, is.null)]
final <- do.call("rbind", res)
write.csv(final, paste0(results_dir, "/max_locs_and_vals.csv"))
write.csv(data.frame("RFE V" = colnames(rfe_import), "Filename" = files[col_keep]),
          file = "a-RFE-30-DISTR-key.csv")
