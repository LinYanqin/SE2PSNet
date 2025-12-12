# 1. Dependencies
	1.1.python == 3.6.3
	1.2.h5py == 2.10.0
	1.3.numpy == 1.19.5
	1.4.torch == 1.10.2
Model has been written and tested with the above dependencies. Performance with other versions has not been tested.
# 2. Steps to use the model to process data
## Step1: Preprocess the original FID (free induction decay). 
Firstly, zero-fill the original FID with zeros to match the network input dimensions. Specifically, the FID containing fewer than 4096 complex points is zero-filled to 4096 complex points. If the FID contains more than 4096 complex points, it should be determined whether the samples beyond 4096 points still contain meaningful signal: if the portion after 4096 points is dominated by noise, only the first 4096 complex points are retained; otherwise, if the signal has not been fully acquired and meaningful signal remains beyond 4096 points, the FID is zero-filled to 8192 or more, i.e., an integer multiple of 4096. Then, the spectrum is divided into two or more spectra, spectra with 4096 points are used as the network input, and the processed spectra are finally spliced to obtain a complete spectrum. Secondly, the Fourier transform is applied to the processed (cropped or padded) FID, the real part is taken, and the result is normalized by its maximum value. Finally, the normalized data (to be processed, as the input of the network) are stored using MATLAB or Python in a “.mat” file (MAT v7.3) with the file name “data.mat”.
## Step 2: Process the 'data.mat' file using the network. 
Run the 'detector.py' file with loading the trained network weight file named 'net.pt' in the 'params' folder to process the data. The output of the network will be saved in the '.mat' format in the 'predict' folder, and can be further visualized through plotting using the 'plot_result.m' file.
# 3. Experimental data

Four experimental data including 'exp_asarone.mat', 'exp_azithromycin.mat', 'exp_estradiol.mat', 'exp_mixture1.mat', 'exp_mixture2.mat', 'exp_mixture3.mat', and 'exp_mixture4.mat' used in our manuscript are provided in the 'exp' folder, which can be used as example data.

