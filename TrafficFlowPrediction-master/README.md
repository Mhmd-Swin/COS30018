# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
tensorflow~=2.16.2
keras~=3.6.0
numpy~=1.26.4
pandas~=2.2.2
matplotlib~=3.9.2
tkcalendar~=1.6.1
pydot~=1.2.4
scikit-learn~=1.5.2
h5py~=3.12.1
geopy~=2.4.1
folium~=0.17.0



**You can use `pip install -r requirements.txt` to install all requirements.**


Run `train.py` to train all models and takes approx. 10 hours on average computer 

Run `data.py` and it will preprocess data for the location 'HIGH STREET_RD W of WARRIGAL_RD'. This generates two files, `data/train.csv` and `data/test.csv`, which will be used in the next step.

`preprocess.py`
**Run command below to train the model:** 

