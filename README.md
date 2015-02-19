To run the model copy the file l_u_soln.py to an appropriate directory and enter:

*ipython l_u_soln.py*

The script will download and unzip the appropriate data files and run the model.


####Some general points:  

1. While building the model, care had to be taken not to encode any forward looking information. For example the column with outstanding loan amount information will be a perfect predictor if a loan has been fully paid. These type of features were not included in the model.  

1. Besides output to stdout there will also be a csv file with results written to disk
