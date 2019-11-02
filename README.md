# Monte Carlo Gradient Estimation in Machine Learning

This is the example code for the following  paper.  If you use the code
here please cite this paper.

> Shakir Mohamed, Mihaela Rosca, Michael Figurnov, Andriy Mnih
  *Monte Carlo Gradient Estimation in Machine Learning*.  [\[arXiv\]](https://arxiv.org/abs/1906.10652).


## Running the code

The code contains:

  * the implementation the score function, pathwise and measure valued estimators `gradient_estimators.py` and their tests to ensure unbiasedness `gradient_estimators_test.py`.
  * the implementation of control variates `control_variates.py` and their tests `control_variates_tests.py`.
  * a `main.py` file to reproduce the Bayesian Logistic regression experiments in the paper.
  * a `config.py` file used to configure experiments.

To run the code and install the required dependencies:

```
  source monte_carlo_gradients/run.sh
```

To run a test:

```
  python3 -m monte_carlo_gradients.gradient_estimators_test
```


## Colab

You can run the code in the browser using [Colab](https://colab.research.google.com). The experiments from Section 3 can be reproduced using the following link: [Intuitive Analysis of Gradient Estimators](https://colab.research.google.com/github/deepmind/mc_gradients/blob/master/monte_carlo_gradients/variance_numerical_integration.ipynb)

## Disclaimer

This is not an official Google product.
