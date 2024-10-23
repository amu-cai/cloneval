# Clone Eval Protocol

## Instructions for Running the Evaluation Script

1. **Download WavLM-Large and save it in the main folder as WavLM-Large.pt.**
   
2. **Split the .wav files into two folders: one for original files and one for cloned files.**

3. **The original and cloned files must have the same name.**

4. **If you want the evaluation protocol to consider emotions, save the .wav files in the format "..._emotion.wav".**

5. **If you run the script without arguments, it will evaluate a sample dataset by default.**

6. **Arguments:**
   - **emotion_list**: A list of emotions in your dataset.
   - **original_dir**: The path to the folder with original samples.
   - **cloned_dir**: The path to the folder with cloned samples.