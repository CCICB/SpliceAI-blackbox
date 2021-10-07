#' Read in DNA from FASTA file
#' @param file A string. The FASTA file location.
ready_fasta <- function(file){
  
  fasta <- readLines(file)
  res <- list()
  for(lineno in 1:length(fasta)){
    line <- fasta[lineno]
    if(grepl(">", line)){
      newlist <- list(NULL)
      active_line <- line
      names(newlist) <- active_line
      append(names, newlist)
    }else{
      res[[active_line]] <- paste0(res[[active_line]], line)
    }
  }
  return(res)
}

#' Ready DNA for SpliceAI model
#' @param seqs A list. Many DNA sequences.
ready_dna_bulk <- function(seqs){
  
  maxx <- max(sapply(seqs, nchar))
  seqs.array <- array(0, c(length(seqs), maxx, 4)) # make an array sized [N x P x 4], fill with zeros
  for(i in 1:length(seqs)){
    seq <- seqs[[i]] # for each sequence
    pos <- factor(unlist(strsplit(seq, "")), levels = c("A", "C", "G", "T")) # convert string into factor
    pos.cat <- to_categorical(as.numeric(pos) - 1, num_classes = 4) # one-hot encode AUGC to a (nchar x 4) matrix
    seqs.array[i,1:nrow(pos.cat),1:4] <- pos.cat # fill [i, , ] with [P x 4] encoding
  }
  return(seqs.array)
}
