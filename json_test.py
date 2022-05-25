import json
import numpy as np 


scores = np.array([69, 420, 666])

data = {}
data['label'] = 3
data['scores'] = scores.tolist()

with open('json_dump_test.txt', 'w') as file:
	json.dump(data, file)

with open('json_dump_test.txt','r') as file:
	loaded_data = json.load(file)


breakpoint()
