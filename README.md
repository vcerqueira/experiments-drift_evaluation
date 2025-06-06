# Drift Evaluation Framework

This repository contains a the experiments concerning a comprehensive framework for evaluating concept drift detection methods based on known drift locations. The experiments also benchmark different state-of-the-art drift detection approaches.

## Project Overview

The framework provides:

- Tools for evaluating concept drift detection methods against ground truth drift locations
- A drift simulation system that can generate different types of concept drift:
  - Abrupt vs gradual drift patterns
  - Covariate drift (changes in input features)
  - Target drift (changes in output variable)
- Benchmarking capabilities for comparing drift detection methods

## Setup & Dependencies

### Requirements
- Python 3.10+
- [CapyMOA library](https://capymoa.org/)

For detailed version requirements of all dependencies, see `requirements.txt`

## Usage

The experiments can be found in the `experiments` folder. 

