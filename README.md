# Valid Oversampling Schemes to Handle Imbalance

This repository provides a Pytorch implementation of the paper "Valid Oversampling Schemes to Handle Imbalance", accepted at *Pattern Recognition Letters*. We compare accuracy, sensitivity, and specificity of various oversampling methods using the MNIST dataset. [[paper link]](https://doi.org/10.1016/j.patrec.2019.07.006).

## Quick start

To execute the valid oversampling method, please go to the directory `lib` first and run the `main.py` file as follows.
```
python3 main.py --experiment 4 # Valid-oversampling method
```

The followings are arguments for other experitments.
```
Experiment 0: No oversampling
Experiment 1: Oversample minor class to be balanced
Experiment 2: Use a cost-sensitive loss function
Experiment 3: SMOTE
Experiment 4: Valid oversampling
```

## Results

|                    	| Accuracy 	| Sensitivity 	| Specificity 	|
|--------------------	|----------	|-------------	|-------------	|
| Naive oversampling 	| .9612    	| .9643       	| .9609       	|
| SMOTE [1]          	| .9828    	| .9398       	| .9875       	|
| Cost-sensitive     	| .9558    	| .9714       	| .9541       	|
| Valid-oversampling 	| .9789    	| .9602       	| .9809       	|


## Reference

[1] Chawla,  N.V.,  Bowyer.,  K.W.,  Hall,  L.O.,  Kegelmeyer,  W.P.,  2002.   Smote:Synthetic  minority  over-sampling  technique.   Journal  of  Artificial  Intelligence Research 16, 321–357.



