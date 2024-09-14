# Molecule Retrosynthesis Prediction using UALIGN Model

This repository contains code, data, and modifications for predicting chemical reactants from product molecules using the **pre-trained UALIGN model**. The project focuses on retrosynthesis analysis, using a **graph-based neural network** to predict reactants from product molecules.

## Objective

The goal of this project is to predict the reactants from given product molecules using machine learning-based retrosynthesis analysis. The project utilizes a **pre-trained UALIGN model** with several custom modifications to improve the performance of retrosynthesis prediction.

## Folder Structure

- **`.ipynb_checkpoints/`**: Checkpoints from Jupyter notebook.
- **`K12343657_Challenge_3.ipynb`**: Main Jupyter notebook running the retrosynthesis prediction process.
- **`modified_graph_utils.py`**: Custom modifications to handle atom mapping, edge and node feature extraction for better graph representation.
- **`modified_inference_one.py`**: Adjustments to the inference script for model parameters, beam search, and data handling.
- **`processed_results.csv`**: Results of reactant predictions based on the test dataset.
- **`Challenge3.pdf`**: Project documentation detailing the approach and results.
- **`K12343657_Molecule_Retrosynthesis.zip`**: Compressed files related to the retrosynthesis project.

## Dataset and Model

### Dataset: USPTO-FULL
- A comprehensive database for chemical reactions used for both training and testing.
- The dataset includes SMILES strings, which represent the molecular structure of compounds in a linear form.

### Model: UALIGN Pre-trained Model
- **UALIGN** is a graph-based neural network that uses **GATBase** (Graph Attention Networks) with a **TransformerDecoder**.
- **Tokenizer**: Custom tokenizer used for the USPTO-FULL dataset.
  
## Key Modifications

### Graph Utility (`modified_graph_utils.py`)
- **Atom Mapping**: Modifications made to handle atom mapping and feature extraction for graph representation of molecular structures.
- **Edge/Node Features**: Adjustments to improve the representation of chemical bonds (edges) and atoms (nodes) in the model.

### Inference Script (`modified_inference_one.py`)
- **Parameters**: Custom settings for model dimensions, layers, and device handling (GPU/CPU).
- **Beam Search**: Adjustments made for better prediction accuracy by refining the beam search technique.
- **Optimizer**: **Adam** optimizer with a learning rate of `0.001` for efficient gradient descent.
  
## Execution

### Prerequisites

To run the project, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- RDKit (for SMILES processing)
- NumPy
- Pandas

### Running the Jupyter Notebook

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Open the **Jupyter Notebook**:
   ```bash
   jupyter notebook K12343657_Challenge_3.ipynb
   ```

3. Run the cells to preprocess the data, load the UALIGN model, and perform the retrosynthesis predictions.

### Data Preprocessing

- **SMILES Strings**: Input SMILES strings are converted into graph objects using the custom graph utilities.
- **Graph Representation**: The molecular graphs are passed into the UALIGN model for reactant prediction.

### Model Inference

- The pre-trained UALIGN model is used to predict reactants from product molecules.
- Predictions are saved in the `processed_results.csv` file, which contains the predicted reactants for each product.

## Results

- The model successfully predicted reactants for a set of product molecules.
- Results were saved in the `processed_results.csv` file.
- **Best Model**: The best-performing model was saved and used for inference after tuning parameters for beam search and model layers.

## Experimentation

### Seq2Seq LSTM Model
- A custom **sequence-to-sequence LSTM** model was developed, but it did not achieve satisfactory accuracy for retrosynthesis prediction.

### Pre-trained BART Model
- A pre-trained **BART model** from Hugging Face was tested, but the performance was similar to the LSTM model, with subpar results in predicting reactants from product molecules.

## Usage

### Running the Inference

To run inference on new data, modify the `inference_one.py` script to point to your dataset and run:
```bash
python modified_inference_one.py
```

This will output the predictions to `processed_results.csv`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The **UALIGN** model was taken from the [UALIGN GitHub repository](https://github.com/zengkaipeng/UAlign).
- Special thanks to the **USPTO-FULL** dataset providers for making the chemical reaction dataset available.