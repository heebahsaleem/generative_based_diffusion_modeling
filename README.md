 The primary objective is to perform conditional generation of geological structures ahead of drilling, when only partial information about the subsurface is available. Using the outpainting strategy with the RePaint
algorithm enables the generation of unknown regions while strictly keeping known information. The same pipeline is applied to both
handwriting and geological datasets to demonstrate the approach’s generalization and enable intuitive comparisons for spatial prediction.
As shown in Figure 1 , the proposed framework first trains a diffusion model on complete realizations. During inference, the known
region is enforced through a binary mask, and the unknown region is generated via conditioned reverse diffusion.
