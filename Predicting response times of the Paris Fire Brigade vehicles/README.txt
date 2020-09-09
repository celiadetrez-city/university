2 supplementary files have been made available to facilitate the exploitation of GPS tracks of emergency vehicles.

 - 1 for the training dataset: x_train_additional_file.csv
 - 1 for the testing dataset: x_test_additional_file.csv

These files are composed by the following data:
 - emergency vehicle selection: identifier of a selection instance of an emergency vehicle for an intervention

 - OSRM estimate from last observed GPS position (json object): service route response from last observed GPS position of an OSRM instance (http://project-osrm.org/docs/v5.15.2/api/#route-service) setup with the Ile-de-France OpenStreetMap data

 - OSRM estimated distance from last observed GPS position (float): distance (in meters) calculated by the OSRM route service from last observed GPS position

 - OSRM estimated duration from last observed GPS position (float): transit delay (in seconds) calculated by the OSRM route service from last observed GPS position

 - time elapsed between selection and last observed GPS position (in seconds)

 - updated OSRM estimated duration (float): time elapsed (in seconds) between selection and last observed GPS position + OSRM estimated duration from last observed GPS position
