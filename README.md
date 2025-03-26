# BlockGuard: Blockchain Malicious Node Detection System

A machine learning-based system for identifying potentially malicious nodes in blockchain networks through transaction pattern analysis and node behavior classification.

## Project Overview

BlockGuard analyzes blockchain transaction data to detect potentially malicious nodes using various machine learning classification algorithms. By examining features such as transaction fees, block generation rates, stake distribution, and node efficiency metrics, the system can identify abnormal behavior patterns that may indicate security threats.

## Key Features

- **Multiple Classification Models**: Implements and compares Random Forest, Support Vector Machine (SVM), and Neural Network (MLP) classifiers
- **Hyperparameter Optimization**: Uses GridSearchCV to find optimal parameters for each model
- **Data Preprocessing**: Includes feature scaling and handling of time-based data
- **Performance Evaluation**: Comprehensive metrics including precision, recall, F1-score, and accuracy
- **Malicious Node Identification**: Flags potentially harmful nodes in the network

## Dataset

The analysis is based on blockchain transaction data (`Blockchain103.csv`) containing various features:

- Block-related features (Height, Generation Rate, Density, Score)
- Transaction features (Fee, Size, Velocity)
- Staking information (Reward, Distribution Rate, Coin Stake)
- Node metrics (Efficiency, Uptime, Latency)
- Node status classification (for training)

## Models Implemented

1. **Random Forest Classifier**
   - Ensemble learning method using multiple decision trees
   - Optimized for depth, estimators, and sample parameters

2. **Support Vector Machine (SVM)**
   - Implements linear and RBF kernels
   - Optimized for regularization parameter (C) and kernel coefficients

3. **Multi-Layer Perceptron (Neural Network)**
   - Deep learning approach with configurable hidden layers
   - Tests multiple activation functions and regularization parameters

## Results

The system compares the performance of all three models and selects the best performer based on accuracy. Detailed classification reports show precision, recall, and F1-score for each class, providing insights into model effectiveness for detecting malicious nodes.

## File Structure

- `model_selection.py`: Script for comparing different ML models and hyperparameter tuning
- `SVM.py`: Implementation focusing specifically on SVM for malicious node detection
- `Blockchain103.csv`: Dataset containing blockchain transaction and node data

## How to Run

1. Clone the repository
   ```
   git clone https://github.com/yourusername/BlockGuard.git
   cd BlockGuard
   ```

2. Install required packages
   ```
   pip install pandas scikit-learn matplotlib numpy
   ```

3. Run the model selection script to compare models
   ```
   python model_selection.py
   ```

4. For specific SVM implementation
   ```
   python SVM.py
   ```

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- numpy
- matplotlib (for visualization, if needed)

## Future Work

- Real-time node monitoring implementation
- Integration with blockchain networks for live detection
- Addition of more advanced models like XGBoost and deep learning architectures
- Development of an alert system for network administrators
- Feature importance analysis to identify key indicators of malicious behavior

## Author

Adithya Sriramoju

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The blockchain community for providing insights on security measures
- Contributors to the scikit-learn library for their excellent ML tools
