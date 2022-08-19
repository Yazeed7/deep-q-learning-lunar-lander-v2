# Project 2 ReadMe

https://github.gatech.edu/storage/user/51271/files/5331b6e4-0027-4698-ae16-d4c1f24396f7

The code provided in [main.py](main.py) replicates the figures used in Project 2's report. It requires the following depedencies:
~~~
python==3.6.13
numpy==1.18.0
matplotlib==3.3.4
gym==1.17.2
pybox2d==2.3.10
tqdm==4.64
pytorch==1.10.2
~~~
Simply run the file and the plots will be generated in the same directory. Note that the creation of these figures will be time consuming, since each figure reguires training at least one Deep Q-Leaning model.
~~~
python main.py
~~~

The plots provided in the report have identical values but with a different style following IEEE standards and are saved as vector images (PDF) instead of PNGs. These were generated using [Lunar_Lander.ipynb](Lunar_Lander.ipynb). However, the setup required is somewhat burdensome, as it requires an installation of LaTeX and configuring matplotlib to use LaTeX as shown in matplotlib's [documentation](https://matplotlib.org/stable/tutorials/text/usetex.html). It additionally requires:
~~~
scienceplots==1.0.9
optuna==2.10 (only to tune hyperparameters)
ffmpeg==4.3.2 (only to generate video)
~~~
