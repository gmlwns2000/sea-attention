### How to Run each implementation
##### 1. Dynamic k Implementation for BERT and OPT Models (Figure 6, Figure 7(b))

This implements dynamic k functionality for BERT and OPT models, as described in Section 4.3 of the official paper. Experiments on Wikitext-2 and MNLI datasets show that increasing k post-training improves accuracy without requiring additional fine-tuning. Relevant results are visualized in Figure 6 and Figure 7(b) of the paper.

Dynamic k enables flexible models that adapt to real-time service demands: reducing k minimizes computational costs (Figures 5(a), 7(a)), while increasing k prioritizes accuracy. Notably, SEA models (k = 32, 64, 128) outperform the vanilla quadratic model (perplexity 29.2 in Figure 6) when k is increased post-training.

**Supported Files:**
- ```src/main/tests/test_dynamic_k_glue.py```: Figure 7(b), BERT on GLUE MNLI dataset
- ```src/main/tests/test_dynamic_k_opt.py```: Figure 6, OPT on Wikitext-2 dataset
Comment out line 1292 in src/models/perlin_attention/attention.py when running each file

##### 2. Validation Curves for SEA and Performer on OpenWebText Dataset (Figure A.5 in Appendix A.9)

This implements a script that generates validation curves comparing SEA and Performer models when trained on the OpenWebText dataset (Figure A.5 in Appendix A.9). The script evaluates and visualizes the convergence performance of both models, showcasing SEA's significantly faster convergence compared to Performer.

**Key Features**
- Validation Curve Visualization:

   - Plots Perplexity (PPL) scores as a function of optimizer steps for SEA and Performer models.
   - Includes a comparison of validation curves highlighting SEA's faster convergence.
- Dataset: OpenWebText dataset

- Run src/main/plot/figure_opt_curve_web.py with ./plots/main/wandb_opt125_openwebtext.csv

##### 3. Visualization Script for Intermediate Buffers in Attention Estimation of BERT (Fig. 10 BERT ver.) 

This implements a script for visualizing the intermediate buffers used during attention estimation in the BERT model. Specifically, this script captures and processes the attention outputs, including masking and sparse interpolation steps and save the metadata. While Figure 10 of the official paper presents visualization results for the OPT model, this script focuses on generating similar visualizations for the BERT model.

- Run src/main/visualize/glue_for_fig10 with saved checkpoints under ./saves/trainer/bert_glue_trainer/
- Comment out line 1292 in src/models/perlin_attention/attention.py

##### 4. Visualization Script for Intermediate Buffers in Attention Estimation (Fig. 10)

This implements a script for visualizing the intermediate buffers used during attention estimation in the OPT model. Specifically, this script captures and processes the attention outputs, including masking and sparse interpolation steps, to generate visualizations similar to those in Figure 10 of the official paper.

**Key Features**
- Visualization of Attention Buffers:

   - Captures intermediate attention scores and visualizes their progression through each decoder layer.
   - Generates layer-wise visualizations for attention patterns in the OPT model.

- Dataset Support:

   - By default, the script works with the wikitext2 dataset but can be configured for other datasets.

- Teacher-Student Model Evaluation:

   - Leverages a teacher-student framework to compare the teacher model's attention scores with the student's attention mechanism.
   - Includes options to evaluate the model and calculate average attention sparsity (k).

- Layer-Wise Attention Visualization:

   - Generates .png images for each decoder layer, showcasing attention patterns and their transformation across layers.

**How to Use**
1. Run the Script: Use the following command to generate visualizations:
```python src.main.visualize.opt_for_fig_10 --dataset <dataset_name> --checkpoint <checkpoint_path> --evaluate```
- --dataset: Dataset to use (default: wikitext2).
- --checkpoint: Path to the model checkpoint for evaluation.
- --evaluate: If specified, evaluates the model and calculates average sparsity (k).
- --max-seq-len: Maximum sequence length for evaluation (default: 2048).

2. Generated Outputs:

- Visualization files are saved in ./plots/visualize_opt/ with subdirectories for each dataset and sample index.
- Each layer's attention visualization is saved as a .png file, named l<layer_number>.png.

**Output Directory**

- Visualizations are saved in ./plots/visualize_opt/<dataset_name>_<sample_index>/.
- Example structure:
```
./plots/visualize_opt/wikitext2_0/
  ├── l0.png
  ├── l1.png
  ├── ...
  └── ln.png 
  ```

##### 5. Test script for Causality Check in OPT model with SEA attention

This introduces a test script for verifying the causality condition in the OPT model using SEA attention. Inspired by the concept of a stack canary, this test injects random inputs sequentially and checks whether the causality condition is violated in the presence of SEA attention. The results confirm that the OPT model with SEA attention satisfies the causality condition effectively.

**Purpose**
The primary goal of this script is to ensure that SEA attention mechanisms maintain the causality condition when applied to the Causal Models. By injecting a specific canary value into the input and comparing it with normal inputs, the script checks for any violations in the outputs of context layers and attention probabilities.

**Key Features**
- generate random hidden states and computes attention outputs using SEA attention.
- check for any discrepancies in causality by comparing outputs with and without injected random inputs.
- handle edge cases and boundary conditions (e.g., canary injection) to ensure robustness.
- Errors (if any) are logged for debugging

**Supported File:**
src/main/tests/test_perlin_opt_causal.py: Implements the causality check for the OPT model with SEA attention.

**How to Run the Test**
```python src/main/tests/test_perlin_opt_causality.py --canary```

##### 6. Test Script for Causality Check in Baseline Attention Methods

This introduces a test script designed to validate whether baseline attention mechanisms satisfy the causality condition.

**Purpose**
The primary goal of this script is to ensure that baseline attention mechanisms maintain the causality condition. By injecting a specific canary value into the input and comparing it with normal inputs, the script checks for any violations in the outputs of context layers and attention probabilities.

**Key Features**
- Causality Validation:

   - Tests various baseline attention mechanisms (e.g., Performer, standard OPTAttention) for compliance with the causality condition.
   - Compares outputs of normal inputs and canary-injected inputs for discrepancies.
- Baseline Method Configuration:

   - The baseline attention method can be specified using the --baseline argument, enabling flexibility for testing various mechanisms.
- Canary Injection:

   - Injects a high "canary" value (300000) at a specific index in the input sequence to validate robustness and adherence to the causality condition.
- Error Logging:

   - Uses a logarithmic error metric to compute discrepancies in the context layer and attention probabilities. Logs any identified causality violations for further analysis.
- Flexible Input Configurations:

   - Supports adjustable parameters like batch size (N), number of attention heads (H), sequence length (T_DST), and hidden dimension size (HID) through the script.

**How to Use**
To run the script, use the following command:

```python src/main/tests/test_baseline_opt_causality.py --baseline <baseline_method> [--canary]```

- --baseline: Specify the baseline attention method to test (default: performer).
- --canary: Enable canary injection to test the robustness of the attention mechanism.

