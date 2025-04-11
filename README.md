# ClonEval

Welcome to **ClonEval**, a framework designed to evaluate and benchmark voice cloning models.

The primary goal of **ClonEval** is to **measure the capability of voice cloning models to correctly capture speaker's voice**. This method operates without human intervention, utilizing detailed analysis of key audio features and WavLM speaker embeddings, allowing users to not only assess the overall quality of a model, but also gain a deeper understanding of how specific aspects of sound influence the final outcome.

## TL;DR

```bash
# Clone this repository to your local machine
git clone https://github.com/amu-cai/cloneval.git
cd cloneval

# Install required packages
pip install -r requirements.txt

# Run our example
python eval.py                   # Without aggregation per emotional state
python eval.py --use_emotion     # With aggregation per emotional state
```

## Usage

1. First, clone this repository to your local machine and install the required packages:

   ```bash
   git clone https://github.com/amu-cai/cloneval.git
   cd cloneval
   pip install -r requirements.txt
   ```

2. Organize the input files into two folders -- one with original samples, and one with cloned ones. Each folder should contain corresponding files with the same filenames for proper comparison. If you want to consider emotional states in the evaluation, make sure that each filename contains name of the relevant emotion after `_`.

   For example:

   ```
   original_samples/
   ├── sample_1_anger.wav
   ├── sample_2_neutral.wav
   └── sample_3_neutral.wav

   cloned_samples/
   ├── sample_1_anger.wav
   ├── sample_2_neutral.wav
   └── sample_3_neutral.wav
   ```

3. Run the `eval.py` script with the following command:

   ```bash
   python eval.py --original_dir <dir_with_original_samples> --cloned_dir <dir_with_cloned_samples> --output_dir <dir_to_save_results>
   ```

   Replace `<dir_with_original_samples>`, `<dir_with_cloned_samples>` and `<dir_to_save_results>` with the paths to your directories. If you want to consider emotions in the evaluation and aggregate the results per emotional state, add the `--use_emotion` flag:

   ```bash
   python eval.py --original_dir <dir_with_original_samples> --cloned_dir <dir_with_cloned_samples> --output_dir <dir_to_save_results> --use_emotion
   ```

4. The script generates two output files in the specified directory:
   - `results.csv` - detailed metrics for each file pair.
   - `aggregated_results.csv` - averaged results for the dataset (per emotion or not).