library(keras) # this does the deep learning stuff
library(caress) # this has some helper functions for deep learning stuff
library(seqLogo) # this makes the PWM
library(tensorflow)
tf$compat$v1$disable_eager_execution() # for backwards compatibility
source("0-global.R") # this has some helper functions for e.g. 1-hot encoding
system("mkdir ../results")
parent_dir <- "../results/A-ascend"
lambda <- 1 # chosen based on prior experiments
steps <- 10 # set to 50 for real experiments!!!!!!!!!!!!!

for(model_id in 1:5){
  
  # Load in the model (downloaded originally from official SpliceAI GitHub)
  m <- keras::load_model_hdf5(paste0("../SpliceAI-master/spliceai/models/spliceai", model_id, ".h5"))
  
  # Set up folder to save output
  results_dir <- paste0(parent_dir, "/model-", model_id, "-lambda-", lambda)
  system(paste0("mkdir ", parent_dir))
  system(paste0("mkdir ", results_dir))
  
  # For each hidden node j in 1...32
  for(j in 1:32){
    
    x <- ready_dna("N")
    loss <- custom_loss(m, "conv1d_38", filter = j, lambda = lambda)
    gradient <- get_layer_gradient(m, loss)
    
    for(step in 1:steps){
      
      x <- caress::ascend(m, x, loss, gradient)
      
      # Save snapshot of 103bp region at steps of 10...
      if(step %% 10 == 0){
        
        p_j <- x[1,4949:5051,] # 103bp region
        
        # Save the prototype
        write.csv(p_j,
                  file = paste0(results_dir, "/filter-", j, "-step-", step, "-prototype.csv")
        )
        
        # Save the pwm
        m_j <- make_pwm(p_j)
        write.csv(m_j,
                  file = paste0(results_dir, "/filter-", j, "-step-", step, "-pwm.csv")
        )
      }
    }
  }
}
