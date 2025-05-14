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
python eval.py --evaluate_emotion_transfer     # With aggregation per emotional state
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

   Replace `<dir_with_original_samples>`, `<dir_with_cloned_samples>` and `<dir_to_save_results>` with the paths to your directories. If you want to consider emotions in the evaluation and aggregate the results per emotional state, add the `--evaluate_emotion_transfer` flag:

   ```bash
   python eval.py --original_dir <dir_with_original_samples> --cloned_dir <dir_with_cloned_samples> --output_dir <dir_to_save_results> --evaluate_emotion_transfer
   ```

4. The script generates two output files in the specified directory:
   - `results.csv` - detailed metrics for each file pair.
   - `aggregated_results.csv` - averaged results for the dataset (per emotion or not).

## Citation Information

You can access the ClonEval paper at arXiv. Please cite the paper when referencing the benchmark, along with the used datasets.

```
@misc{christop2025clonevalopenvoicecloning,
      title={ClonEval: An Open Voice Cloning Benchmark}, 
      author={Iwona Christop and Tomasz Kuczyński and Marek Kubis},
      year={2025},
      eprint={2504.20581},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.20581}, 
}

@article{crema-d,
    author={Cao, Houwei and Cooper, David G. and Keutmann, Michael K. and Gur, Ruben C. and Nenkova, Ani and Verma, Ragini},
    journal={IEEE Transactions on Affective Computing},
    title={{CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset}},
    year={2014},
    volume={5},
    number={4},
    pages={377--390},
    doi={10.1109/TAFFC.2014.2336244},
}

@inproceedings{librispeech2015,
    author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
    booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={{Librispeech: An ASR corpus based on public domain audio books}}, 
    year={2015},
    pages={5206-5210},
    keywords={Resource description framework;Genomics;Bioinformatics;Blogs;Information services;Electronic publishing;Speech Recognition;Corpus;LibriVox},
    doi={10.1109/ICASSP.2015.7178964}
}

@article{ravdess,
    doi={10.1371/journal.pone.0196391},
    author={Livingstone, Steven R. AND Russo, Frank A.},
    journal={PLOS ONE},
    publisher={Public Library of Science},
    title={{The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English}}",
    year={2018},
    month=may,
    volume={13},
    URL={https://doi.org/10.1371/journal.pone.0196391},
    pages={1--35},
    number={5},
}

@inbook{savee,
    author={Haq, S. and Jackson, P. J. B.},
    booktitle={{Machine Audition: Principles, Algorithms and Systems}},
    title={{Multimodal Emotion Recognition}},
    publisher={IGI Global},
    address={Hershey PA},
    year={2010},
    month=aug,
    editor={Wang, W.},
    pages={398--423},
}

@misc{tess,
    author={Pichora-Fuller, M. Kathleen and Dupuis, Kate},
    publisher={Borealis},
    title={{Toronto emotional speech set (TESS)}},
    year={2020},
    version={DRAFT VERSION},
    doi={10.5683/SP2/E8H2MF},
    URL={https://doi.org/10.5683/SP2/E8H2MF},
}
```

