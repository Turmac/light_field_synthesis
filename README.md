# Synthesizing Light Field From a Single Image with Variable MPI and Two Network Fusion (SIGGRAPH Asia 2020)

This is the code for the SIGGRAPH Asia 2020 paper.  
Paper page: https://people.engr.tamu.edu/nimak/Papers/SIGAsia2020_LF/index.html

<div style="align:center">
    <img src='https://user-images.githubusercontent.com/5975007/97067709-e3b42600-1585-11eb-9b53-90e405f3e0d8.gif'/>
</div>

## Environment
This code has been tested under Windows 10, Python 3.7.7, CUDA 10.1  
Required libraries:  
<pre>
matplotlib            3.2.1  
numpy                 1.16.6  
opencv-python         4.2.0.34  
Pillow                7.1.2  
scikit-image          0.17.2  
scikit-learn          0.23.1  
scipy                 1.4.1  
tensorboardX          2.0  
torch                 1.5.0+cu101  
torchfile             0.1.0  
torchvision           0.6.0+cu101  
</pre>

## Usage
1. Download the testting data at https://drive.google.com/drive/folders/1hsHruwHIEuVQWfSlrgOSGlPrruWHdwqm?usp=sharing
2. Extract the data under the data folder
3. Run:
   > python light_field_synthesis.py
4. Check the result:
   > cd results
   > python npy2lf_video8.py

The code can synthesize from 2D image, or synthesize 15 by 15 light field with minor change.  
To test on your data, please use the Deeplens model to predict the depth:  
https://github.com/scott89/deeplens_eval

To train the model, please download:  
Standford Light field dataset: http://lightfields.stanford.edu/mvlf/  
Our Light field dataset: Coming soon  

<h1>Citation</h1>
If this work is helpful in your research. Please cite:  

```
@inproceedings{Li2020LF,
    author = {Li, Qinbo and Khademi Kalantari, Nima},
    title = {Synthesizing Light Field From a Single Image with Variable MPI and Two Network Fusion},
    journal = {ACM Transactions on Graphics}, 
    volume = {39}, 
    number = {6}, 
    year = {2020}, 
    month = {12}, 
    doi = {10.1145/3414685.3417785} 
}
```
