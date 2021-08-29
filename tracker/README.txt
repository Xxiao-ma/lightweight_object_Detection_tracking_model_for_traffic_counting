This folder contains all the files related to the tracking part
1. The main program of the tracking logic is the 'Countin_test_submitt_version.py'
	You can run the tracker by using 'python Countin_test_submit_version.py' 

2. The tracker can be configured by following the instructions in the file 'Countin_test_submit_version.py', by setting config parameters inside.

3. The folder 'newAddedData20210701' contains all the input sequences for the detection/tracker task
4. The folder 'newAddedData20210701_inference' contains the inference results by the best performer detection model 'SSDlite modified MobileNetv2 + add multi background subtraction at stage 3,4'.
5. The tracker use the inference results from above, and then handel the tracking task
6. The folder 'count_results_of_experiments' contains all the tracking results in the tracker experiments.
