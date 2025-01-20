import os
from paths import da
test_sem_file = os.path.join(dataset_path, 'splits', 'nvs_sem_train.txt')
test_semantic = []
with open(test_sem_file) as file:
    test_semantic = [line.rstrip() for line in file]

print(test_semantic)


