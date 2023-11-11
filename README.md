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
> Jorge S. S. Junior, Jerome Mendes, Francisco Souza, and Cristiano Premebida. Hybrid LSTM-Fuzzy System to Model a Sulfur Recovery Unit. In Proc. of the 20th International Conference on Informatics in Control, Automation and Robotics (ICINCO 2023), pages XX-XX, Rome, Italy, November 13th-15th, 2023. DOI: http://doi.org/XX.XXXX/XXXXX.XXXX.XXXXXX

> [!NOTE]
> Some descriptions will be updated in the future for better clarification.
