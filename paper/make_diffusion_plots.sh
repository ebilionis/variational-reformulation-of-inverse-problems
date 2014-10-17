# Author: Ilias Bilionis
# Date:   10/1/2014
# 
# Makes the plots for the diffusion problems

# Make the plots for the first diffusion case (observations on four corners)
python examples/make_diffusion_plots.py \
        --var-file=examples/diffusion_model_num_comp=1.pcl \
        --mcmc-file=examples/diffusion_model_mcmc.h5 \
        --skip=100

# Make the plots for the second diffusion case (observations on the top and bottom centers)
python examples/make_diffusion_plots.py \
        --var-file=examples/diffusion_upperlowercenters_model_num_comp=2.pcl \
        --mcmc-file=examples/diffusion_upperlowercenters_model_mcmc.h5 \
        --max-true-pxi=10 \
        --skip=100
