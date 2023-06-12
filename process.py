import numpy as np
# load tain_accs from np file
test_accs = np.load("final_test_acc.npy")
worst_accs = np.load("worst_test_acc.npy")

results_dict = []
latex_string = ''


for i in range(len(test_accs)):
    if i % 3 == 0:
        results = {'results_mean': np.mean(test_accs[i:i+3])*100, 'results_sdv': np.std(
            test_accs[i:i+3])*100, 'worst_mean': np.mean(worst_accs[i:i+3])*100, 'worst_sdv': np.std(worst_accs[i:i+3])*100}
        latex_string = latex_string+str(round(results['results_mean'],2))+'$\pm$'+str(round(results['results_sdv'],1))+'&'+str(round(results['worst_mean'],2))+'$\pm$'+str(round(results['worst_sdv'],1))+'&'
        # wirte a dictinary that synthesis the mean and sdv
        results_dict.append(results)

print(latex_string)
# 60.83$\pm$1.6 & 24.08$\pm$2.9 & 52.95$\pm$0.6 & 7.73$\pm$0.8 & 69.08$\pm$1.5 & 42.8$\pm$1.9 & 56.03$\pm$1.2 & 15.81$\pm$1.9 & 75.59$\pm$1.3 & 57.84$\pm$1.1 & 59.39$\pm$1.1 & 26.43$\pm$2.1 & 61.28$\pm$0.0 & 28.24$\pm$0.0&
