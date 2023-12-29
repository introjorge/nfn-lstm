# nfn_lstm

Toolbox for implementing a hybrid method using a neo-fuzzy neuron system with long short-term memory, the NFN-LSTM model.

## How to Run:
Run the main file "main.py",preferably in a python IDE, to visualize the estimated output and error metrics.

## Other Functions:
- "nfn_aux.py:" auxiliary code to load the dataset to be trained and validated, as well as functions to estimate the antecedent parameters of fuzzy rules and the NRMSE (Normalized Root Mean Squared) error metric.
- "nfn_learning.py:" code to initialize the NFN-LSTM model, train it based on the parameters defined in "main.py" and compare the results with real data.

## Dataset:
- **Task:** Estimation of residual H2S in a sulfur recovery unit[^1].

[^1]: Fortuna, L., Graziani, S., Rizzo, A., & Xibilia, M. G. (2007). Soft sensors for monitoring and control of industrial processes (Vol. 22). London, UK:: Springer. DOI: https://doi.org/10.1007/978-1-84628-480-9

> [!IMPORTANT]
> In case of publication of any application of this method, cite the work:
> 
> J. S. S. Junior, J. Mendes, F. Souza, and C. Premebida (2023). Hybrid LSTM-Fuzzy System to Model a Sulfur Recovery Unit. In Proceedings of the 20th International Conference on Informatics in Control, Automation and Robotics - Volume 2: ICINCO. SciTePress, pages 281-288. DOI: 10.5220/0012165100003543
