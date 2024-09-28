import os
import shutil
import subprocess
from pathlib import Path

def clear_directory(saves_dir,  positive_dir):
   for file in os.listdir(saves_dir):
      pos_file_path = Path(positive_dir) / file
      saves_file_path = Path(saves_dir) / file
      if pos_file_path.exists():
         saves_file_path.unlink()

def opencv_annotation(saves_dir, annotations_file):
   command = [
      r'C:\Users\wuwsh\opencv\build\x64\vc15\bin\opencv_annotation.exe',
      f'--annotations={annotations_file}',
      f'--images={saves_dir}'
   ]
   process = subprocess.Popen(command, text=True)
   process.wait()

def move_files(source_dir, target_dir):
   for file in os.listdir(source_dir):
      shutil.move(Path(source_dir) / file, Path(target_dir))

def generate_positive(pos_file, pos_saves_file):
   with open(pos_saves_file, 'r') as pos_saves:
      saves = pos_saves.readlines()
      saves = [saves[i].replace('saves', 'positive') for i in range(len(saves))]
   with open(pos_file, 'a') as pos:
      pos.write('\n')
      pos.writelines(saves)
   with open(pos_saves_file, 'w') as pos_saves:
      pos_saves.write('')

def main():
    positive_dir = Path('positive')
    pos_file = 'pos.txt'
    saves_dir = Path('saves')
    pos_saves_file = 'pos_saves.txt'

    if saves_dir.exists() and any(saves_dir.iterdir()):
        clear_directory(saves_dir, positive_dir)
        opencv_annotation(saves_dir, pos_saves_file)
        move_files(saves_dir, positive_dir)
        generate_positive(pos_file, pos_saves_file)

if __name__ == "__main__":
    main()