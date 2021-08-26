from datahandler import Datahandler
from os.path import isfile, isdir, join
from os import listdir
import pathlib
import subprocess
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

class Preprocess:


	def __init__(self, datahandler):
		self.dh = datahandler
		self.name = self.dh.name
		
		
	def preprocess(self, preprocess=False, cpu=False):
	
		#Preprocess the data
		
		path_unprocessed = self.dh.get_unprocessed_folder()
				
		path_output = self.dh.get_data_path("home")
		
		if self.name == 'brain_t1':
			path_output = join(path_output,"T1w")
		elif self.name == 'brain_t2':		
			path_output = join(path_output,"T2w")
		else:
			path_output = join(path_output, "neck_brain_cancer")
		
		path_preprocessed = join(path_output, "imgs")
		path_results = join(path_output, "results")
		path_masks = join(path_output, "masks")
		
		#create necessary directories
		
		pathlib.Path(path_preprocessed).mkdir(parents=True, exist_ok=True)
		pathlib.Path(path_results).mkdir(parents=True, exist_ok=True)
		pathlib.Path(path_masks).mkdir(parents=True, exist_ok=True)

		
		skull_stripping_input = path_unprocessed
		skull_stripping_output = path_results
		
		affine_input = path_unprocessed
		affine_output = path_preprocessed
		
		normalize_input = path_preprocessed
		normalize_output = path_results
		
		false_ordering = False
		
		if preprocess:
			false_ordering = True
			print("start preprocessing")

			
			process = subprocess.run(["preprocess -i " + path_unprocessed + " -o " + path_output], check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			
			print(process.stdout)

		
			skull_stripping_input = path_preprocessed
			affine_input = path_preprocessed
			affine_output = path_results
			normalize_input = path_results
			normalize_output = path_preprocessed
		
		if self.name in ['brain_t1', 'brain_t2']:
			false_ordering = False
			self.skull_stripping(skull_stripping_input, skull_stripping_output, path_masks, cpu)
			affine_input = path_results
			affine_output = path_preprocessed
			normalize_input = path_preprocessed
			normalize_output = path_results

		# affine align
		print("affine alignement")
		
		if not false_ordering:
		
			#clear preprocessed
			
			process = subprocess.run(["rm -r " + path_preprocessed], check=True,stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			
			process = subprocess.run(["mkdir " + path_preprocessed], check=True,stdout=subprocess.PIPE, universal_newlines=True, shell=True)
		
			
		process = subprocess.run(["coregister -i " + affine_input + " -o " + affine_output], check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
		
		print(process.stdout)
		
				
		if not false_ordering:
			#clear results
			
			process = subprocess.run(["rm -r " + path_results], check=True,stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			
			process = subprocess.run(["mkdir " + path_results], check=True,stdout=subprocess.PIPE, universal_newlines=True, shell=True)

		else:		
			#clear preprocessed
			
			process = subprocess.run(["rm -r " + path_preprocessed], check=True,stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			
			process = subprocess.run(["mkdir " + path_preprocessed], check=True,stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			
		
		#normalize
		print("nyul normalize")
		process = subprocess.run(["nyul-normalize -i " + normalize_input + " -o " + normalize_output], check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
		
		print(process.stdout)
				
		if false_ordering:
			#swap file names
			temp = join(path_output, "imgs")
			os.rename(path_preprocessed, temp)
			os.rename(temp, path_results)
			os.rename(path_results, path_preprocessed)
		
	
		
	def skull_stripping(self, skull_stripping_input, skull_stripping_output, path_masks, cpu=False):
	
		#Skull stripping the data
		print("skull stripping")
		
		
		if cpu:
			process = subprocess.run(["hd-bet -i " + skull_stripping_input + " -o " + skull_stripping_output + " -device cpu -mode fast -tta 0"], check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
		else:
			process = subprocess.run(["hd-bet -i " + skull_stripping_input + " -o " + skull_stripping_output], check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			
		print(process.stdout)
		
		
		# Move masks to seperate directory
		print("extract masks")
		
		masks = [join(skull_stripping_output, f) for f in listdir(skull_stripping_output) if isfile(join(skull_stripping_output, f)) and "mask" in str(f)]
		
		mask_string = ""
		
		for m in masks:
			mask_string += " " + m
		print(mask_string)
		
		if len(mask_string) != 0:
			process = subprocess.run(["mv -t " + path_masks + " " + mask_string], check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
			print(process.stdout)
		
			
	
	
	
		

