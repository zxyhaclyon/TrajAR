# TrajAR

## Title
TrajAR: Long-Term Trajectory Prediction at Urban Intersections via Multi-scale Interaction Perception

## Data sets
For model training and evaluation, the [SinD](https://github.com/SOTIF-AVLab/SinD/tree/main) and [inD](https://www.ind-dataset.com/) are used. In our experiments, we use four real-world datasets: SinD-Tianjin, SinD-Xian, InD-Bendplatz, and InDFrankenburg. The two SinD datasets record signalized intersections in Tianjin and Xian, China, at 10 Hz. The two InD datasets contain recorded trajectories of four types of road users from different intersections in Germany, with several hours recorded at 25 Hz.

## Preprocessing

Assuming that you have been granted access to any of the above-mentioned data sets, proceed by moving the unzipped content (folder) into a folder named `SinD_data/{Tianjin Xian}` or `inD_data/{location1 location2}` on the same level as this project. 

Methods of preprocessing are contained within Python scripts. Executing them may be done from a terminal or IDE of choice **(from within this project folder)**, for example: 
```bash
python SinD_preprocess.py
python inD_preprocess.py
```

You need to modify the three parameters `city`(only SinD need), `data_read` and `data_save` according to the requirements. When preprocessing the inD, comment out line 294 or 295 in `inD_preprocess.py` to read the files at different intersections. The output of the preprocessing scripts will be sent to a sub-folder with the name of the data set within the `./data` folder in this project.

## Usage

All the architectures of the TrajAR model are included in `Model.py`.  
Different dataset training parameters correspond to different `parser.py` files in `./config` folder. Modify the `InitArgs` class on line 206 of main.py to correspond to different datasets.
```bash
python main.py --ctx
```


