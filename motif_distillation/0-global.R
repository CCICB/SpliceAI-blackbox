#' Convert DNA into 1-hot representation with Ns
#' @param sequence A character. One DNA sequence.
one_hot_dna_N <- function(sequence){
  
  chars <- unlist(strsplit(sequence, ""))
  factors <- factor(chars, levels = c("A", "C", "G", "T", "N"))
  onehot <- keras::to_categorical(as.numeric(factors)-1, 5)
  if(length(dim(onehot)) == 1){
    onehot <- t(onehot)
  }
  onehot <- onehot[,-5,drop=FALSE] # drop N altogether
  #onehot[rowSums(onehot) == 0,] <- .25 # treat missing as .25
  onehot
}

#' Ready DNA for SpliceAI model
#' @param sequence A character. One DNA sequence.
ready_dna <- function(sequence, pad = 5000){
  
  x <- one_hot_dna_N(sequence)
  nullmat <- matrix(0, pad, 4)
  x. <- do.call("rbind", list(nullmat, x, nullmat))
  dim(x.) <- c(1, dim(x.))
  x.
}

make_pwm <- function(proto){
  
  zz <- apply(proto, 1, function(x) exp(x)/(sum(exp(x)))) # as pwm
  zz[is.nan(zz)] <- 1 # NaN due to exp(x) -> Inf
  zz[,colSums(zz) == 2] <- 1/2 # sometimes you get 2x Inf
  zz[,colSums(zz) == 3] <- 1/3 # sometimes you get 3x Inf
  zz[,colSums(zz) == 4] <- 1/4 # sometimes you get 4x Inf
  rownames(zz) <- c("A", "C", "G", "T")
  return(zz)
}

make_logo <- function(proto){
  
  zz <- make_pwm(proto)
  obj <- makePWM(zz)
  seqLogo(obj, ic.scale = TRUE, yaxis = FALSE, xaxis = FALSE)
}

custom_loss <- function(model, layer, filter = NA, lambda = 1){
  
  activity <- get_layer(model, name = layer)$output
  loss <- activity[,5001,filter]
  return(
    loss - lambda * k_sum(k_abs(get_layer(model, "input_1")$output))
  )
}
