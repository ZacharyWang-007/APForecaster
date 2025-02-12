# APForecaster
 This GitHub repository is for the under-review paper "Climate Change Impact on Future Landscape Fire Air Pollution and Global Mortality Burden", particularly for the model development part. 

## Pipeline
<div align="center">
  <img src="Figure.png">
 </div>

## Requirements
### Installation
 Please refer to the requirements.txt file. All the packages as well as versions have been provided.

### Data Preparation
 Please refer to the manuscript for details. Here, we only provide data for casual model testing. [Google Drive](https://drive.google.com/drive/folders/1TgQBj3_peZXYDywGBCg33yHi_z79eKD9?usp=sharing).

## Model Training and Testing
 Before training and testing, please update the configs including args.targets, args.path-indicators, args.path-pm25-o3, args.land and args.mask.
 Generally, we train the model with three 40 GB memory GPUs for about 23 hours on data from 2000 to 2014. 
 ~~~~~~~~~~~~~~~~~~
   python main.py 
 ~~~~~~~~~~~~~~~~~~

## Trained Models and Demo datasets
The well-trained APForecaster models for PM2.5 and O3 can be found in [Google Drive](https://drive.google.com/drive/folders/1TgQBj3_peZXYDywGBCg33yHi_z79eKD9?usp=sharing).

## Contact
If you have any questions regarding the model training and testing, please feel free to contact us. E-mail: [wenhua.yu@monash.edu](zhikang.wang@monash.edu), [zhikang.wang@monash.edu](zhikang.wang@monash.edu) 
