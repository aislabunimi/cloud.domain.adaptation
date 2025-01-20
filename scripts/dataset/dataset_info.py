import json
import os

from scripts.paths import dataset_path

train_sem_file = os.path.join(dataset_path, 'splits', 'nvs_sem_train.txt')
train_semantic = []
with open(train_sem_file) as file:
    train_semantic = [line.rstrip() for line in file]

val_sem_file = os.path.join(dataset_path, 'splits', 'nvs_sem_val.txt')
val_semantic = []
with open(val_sem_file) as file:
    val_semantic = [line.rstrip() for line in file]

test_sem_file = os.path.join(dataset_path, 'splits', 'sem_test.txt')
test_semantic = []
with open(test_sem_file) as file:
    test_semantic = [line.rstrip() for line in file]

environment_types_file = os.path.join(dataset_path, 'metadata', 'scene_types.json')
environment_types = json.load(open(environment_types_file))

print(f'Categories of environment: {len(set(environment_types.values()))}\n'
      f'{sorted(list(set(environment_types.values())))}')
print(f'TRAIN:\n'
      f'Number of environments: {len(train_semantic)}\n'
      f'Has semantics: {len([e for e in train_semantic if os.path.exists(os.path.join("/media/antonazzi/Elements/scannetppv2", "data", e, "scans", "segments_anno.json"))])}\n'
      
      f'Number of apartments: {len([e for e in train_semantic if "apartment" in environment_types[e]])}\n'
      f'Number of offices: {len([e for e in train_semantic if "office" in environment_types[e]])}\n')

print(f'VALIDATION:\n'
      f'Number of environments: {len(val_semantic)}\n'
      f'Has semantics: {len([e for e in val_semantic if os.path.exists(os.path.join("/media/antonazzi/Elements/scannetppv2", "data", e, "scans", "segments_anno.json"))])}\n'

      f'Number of apartments: {len([e for e in val_semantic if environment_types[e] == "apartment"])}\n'
      f'Number of offices: {len([e for e in val_semantic if "office" in environment_types[e]])}\n')

print(f'TEST:\n'
      f'Number of environments: {len(test_semantic)}\n'
      f'Has semantics: {len([e for e in test_semantic if os.path.exists(os.path.join("/media/antonazzi/Elements/scannetppv2", "data", e, "scans", "segments_anno.json"))])}\n'

      f'Number of apartments: {len([e for e in test_semantic if environment_types[e] == "apartment"])}\n'
      f'Number of offices: {len([e for e in test_semantic if "office" in environment_types[e]])}\n')
