# Mamba-UIE: Enhancing Underwater Images with Physical Model Constraint
In underwater image enhancement (UIE), convolutional neural networks (CNN) have inherent limitations in modeling long-range dependencies and are less effective in recovering global features. While Transformers excel at modeling long-range dependencies, their quadratic computational complexity with increasing image resolution presents significant efficiency challenges. Additionally, most supervised learning methods lack effective physical model constraint, which can lead to insufficient realism and overfitting in generated images. To address these issues, we propose a physical model constraint-based underwater image enhancement framework, Mamba-UIE. Specifically, we decompose the input image into four components: underwater scene radiance, direct transmission map, backscatter transmission map, and global background light. These components are reassembled according to the revised underwater image formation model, and the reconstruction consistency constraint is applied between the reconstructed image and the original image, thereby achieving effective physical constraint on the underwater image enhancement process. To tackle the quadratic computational complexity of Transformers when handling long sequences, we introduce the Mamba-UIE network based on linear complexity state space models (SSM). By incorporating the Mamba in Convolution block, long-range dependencies are modeled at both the channel and spatial levels, while the CNN backbone is retained to recover local features and details. Extensive experiments on three public datasets demonstrate that our proposed Mamba-UIE outperforms existing state-of-the-art methods, achieving a PSNR of 27.13 and an SSIM of 0.93 on the UIEB dataset.

![image](https://github.com/user-attachments/assets/a9c2697f-033b-4f5f-be2f-8a004f8af680)

## Compared to other methods
![image](https://github.com/user-attachments/assets/e91ae8ed-2374-4845-bee1-64e99b243cdf)

## Train the Model
python main.py

## Test the Model
python eval.py

## Environment
environment.txt

