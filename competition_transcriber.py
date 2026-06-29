from huggingface_hub import hf_hub_download

import subprocess
import tempfile
import os
import PIL.Image
import malti.line_joiner
import pytesseract

#pip install pytesseract

class CompetitionTranscriber:

    def __init__(self) -> None:
        self.line_joiner = malti.line_joiner.RBLineJoiner()
        # Ensure this path matches the output of 'which tesseract' in your terminal
        self.tesseract_cmd = '/usr/local/bin/tesseract'#'/opt/homebrew/bin/tesseract'
        # Get the path to this directory
        self.traineddata_path = hf_hub_download(
            repo_id="AlanaBusu/DocEng_Tessarect_mlt",
            filename="finetune_tess_extended_2000.traineddata",
        )
        self.tessdata_dir = os.path.dirname(self.traineddata_path)
        #self.model_dir = os.path.dirname(self.traineddata_path)
        

    def transcribe(self, image: PIL.Image) -> str:
        with tempfile.TemporaryDirectory() as path:
            image_path = os.path.join(path, 'img.jpg')
            output_base = os.path.join(path, 'out')
            image.save(image_path)
            
            # Setup environment to point to your local model file
            env = os.environ.copy()
            env['TESSDATA_PREFIX'] = self.tessdata_dir
            
            # Run Tesseract using the local finetune_tess model
            result = subprocess.run(
                [
                    self.tesseract_cmd,
                    "--tessdata-dir",
                    self.tessdata_dir,
                    "-l",
                    "finetune_tess_extended_2000", 
                    image_path,
                    output_base,
                   
                   # '-l', 'finetune_tess_extended_2000',  # Loads 'finetune_tess.traineddata'
                ],
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("--- Tesseract Failed ---")
                print("Stderr:", result.stderr)
                print("Stdout:", result.stdout)
                raise RuntimeError("Tesseract failed to process the image.")
                
            output_file = output_base + '.txt'
            
            with open(output_file, encoding='utf-8') as f:
                text = self.line_joiner.join_lines(
                    f.read().strip().split('\n'), 
                    fix_hyphenated_words=True
                )
                
        return text