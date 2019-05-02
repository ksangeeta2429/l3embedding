# EdgeL3

In EdgeL3 [1], we comprehensively explored the feasibility of compressing the Look, Listen and Learn (L3-Net) audio model for mote-scale inference. We used pruning, ablation, and knowledge distillation techniques to show that the originally proposed L3-Net architecture is substantially overparameterized, not only for Audio Visual Correspondence task but for the target task of sound classification as evaluated on two popular downstream datasets, US8K and ESC50. EdgeL3, a 95% sparsified version of L3-Net, provides a useful reference model for approximating L3 audio embedding for transfer learning.

For the pre-trained embedding models (edgeL3), please go to: [https://github.com/ksangeeta2429/edgel3](https://github.com/ksangeeta2429/edgel3)

Dependencies
* Python 3 (we use 3.6.3)
* [ffmpeg](http://www.ffmpeg.org)
* [sox](http://sox.sourceforge.net)
* [TensorFlow](https://www.tensorflow.org/install/) (follow instructions carefully, and install before other Python dependencies)
* [keras](https://keras.io/#installation) (follow instructions carefully!)
* Other Python dependencies can by installed via `pip install -r requirements.txt`

Note that the metadata format expected is the same used in [AudioSet](https://research.google.com/audioset/download.html) ([Gemmeke, J., Ellis, D., et al. 2017](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf)), as training this model on AudioSet was one of the goals for this implementation.

You can train an AVC/embedding model using `train.py`. Run `python train.py -h` to read the help message regarding how to use the script.

There is also a module `classifier/` which contains code to train a downstream classifier with the embedding extracted from the L3 audio model. In EdgeL3, we evaluated the embedding on two datasets, US8K and ESC50. Run `python train_classifier.py -h` to read the help message regarding how to use the script to train a dowwnstream classifier.
