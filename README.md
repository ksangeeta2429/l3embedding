# EdgeL3

In EdgeL3 [1], we comprehensively explored the feasibility of compressing the Look, Listen and Learn [3] (L3-Net) audio model for mote-scale inference. We used pruning, ablation, and knowledge distillation techniques to show that the originally proposed L3-Net architecture is substantially overparameterized, not only for Audio Visual Correspondence task but for the target task of sound classification as evaluated on two popular downstream datasets, US8K and ESC50. EdgeL3, a 95% sparsified version of L3-Net, provides a useful reference model for approximating L3 audio embedding for transfer learning.

For the pre-trained embedding models (edgeL3), please go to: [https://github.com/ksangeeta2429/edgel3](https://github.com/ksangeeta2429/edgel3)

Dependencies
* Python 3 (we use 3.6.3)
* [ffmpeg](http://www.ffmpeg.org)
* [sox](http://sox.sourceforge.net)
* [TensorFlow](https://www.tensorflow.org/install/) (follow instructions carefully, and install before other Python dependencies)
* [keras](https://keras.io/#installation) (follow instructions carefully!)
* Other Python dependencies can by installed via `pip install -r requirements.txt`

Note that the metadata format expected is the same used in [AudioSet](https://research.google.com/audioset/download.html) ([Gemmeke, J., Ellis, D., et al. 2017](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf)), as training this model on AudioSet was one of the goals for this implementation.

# Using EdgeL3 Repository:

The `l3embedding/pruning.py` has most of the logic to l3 pruning and re-training of the pruned models.</br>

1. Pruning can be done in two ways:</br>
  a. Layerwise with a specified sparsity list.</br>
  b. Filterwise where entire filters (feature maps) are dropped according to the specified sparsity values.</br>
2. After pruning, re-training can be done in two ways:</br>
  a. Fine-tuning the sparsified weights of the audio model by feezing the vision model.</br>
  b. Training the sparsified audio model in teacher-student setting with the original L3 model as teacher.</br>

 Original L3 model:</br>
 
 Original i.e. non-pruned L3 model can be found in `models/cnn_l3_melspec2_recent`. There are two versions:</br>
 a. `model_best_valid_accuracy.h5`, single GPU format - Use its path for WEIGHT_PATH in script files.</br>
 b. `model_best_valid_accuracy_multiGPU.h5`, Multi GPU format - All the L3 models are trained (or re-trained) with 4 GPUS and gpu_wrapper_4gpus decorator in pruning.py is for the same.</br>

There is also a module `classifier/` which contains code to train a downstream classifier with the embedding extracted from the pruned L3 audio model. In EdgeL3, we evaluated the embedding on two datasets, US8K and ESC50. </br>

If you use a SLURM environment, sample sbatch scripts for re-training pruned models are available in jobs/. A sample command to run the script is also mentioned in form of a comment. Note that some of the variables need to be changed depending on your paths.

# References

Please cite the following papers when using EdgeL3 in your work:

[1] **EdgeL3: Compressing L3-Net for Mote-Scale Urban Noise Monitoring** </br>
Sangeeta Kumari, Dhrubojyoti Roy, Mark Cartwright, Juan Pablo Bello, and Anish Arora. </br>
Parallel AI and Systems for the Edge (PAISE), Rio de Janeiro, Brazil, May 2019.

[2] **Look, Listen and Learn More: Design Choices for Deep Audio Embeddings** </br>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[3] **Look, Listen and Learn**<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.

