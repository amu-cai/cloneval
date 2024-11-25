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

# Create ./checkpoints/ directory
mkdir checkpoints
```

Download **WavLM Large** checkpoint from [here](https://github.com/microsoft/unilm/tree/master/wavlm) and save it as `./checkpoints/WavLM-Large.pt`.

Run our example:

```bash
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

2. Ensure there is a directory named `checkpoints` in the project root. If it doesn't exist, create it:

   ```bash
   mkdir checkpoints
   ```

3. Download the required **WavLM Large** checkpoint from [here](https://github.com/microsoft/unilm/tree/master/wavlm) and save it as `WavLM-Large.pt` in the `checkpoints` directory.

4. Organize the input files into two folders -- one with original samples, and one with cloned ones. Each folder should contain corresponding files with the same filenames for proper comparison. If you want to consider emotional states in the evaluation, make sure that each filename contains name of the relevant emotion after `_`.

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

5. Run the `eval.py` script with the following command:

   ```bash
   python eval.py --original_dir <dir_with_original_samples> --cloned_dir <dir_with_cloned_samples>
   ```

   Replace `<dir_with_original_samples>` and `<dir_with_cloned_samples>` with the paths to your directories. If you want to consider emotions in the evaluation and aggregate the results per emotional state, add the `--use_emotion` flag:

   ```bash
   python eval.py --original_dir <dir_with_original_samples> --cloned_dir <dir_with_cloned_samples> --use_emotion
   ```

6. The script generates two output files in the current directory:
   - `results.csv` - detailed metrics for each file pair.
   - `aggregated_results.csv` - averaged results for the dataset (per emotion or not).