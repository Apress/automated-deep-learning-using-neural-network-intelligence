from ch5.naive_one_shot_nas.pt.pt_lenet_multi_model import PtLeNetMultiTrialModel
from ch5.naive_one_shot_nas.pt.pt_train import train_model, test_model

# Search Space
kernel1_choices = [1, 3, 5]
kernel2_choices = [1, 3, 5]

results = {}
# Performing Multi-trial Search
for k1 in kernel1_choices:
    for k2 in kernel2_choices:
        # Trial
        model = PtLeNetMultiTrialModel(k1, k2)
        train_model(model)
        accuracy = test_model(model)
        results[(k1, k2)] = accuracy

# Displaying Results
print('=======')
print('Results:')
for k, v in results.items():
    print(f'Conv1 {k[0]}x{k[0]}, Conv2: {k[1]}x{k[1]} : {v}')
