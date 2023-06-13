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

print("latex_string",latex_string,len(test_accs))

#save string to txt
with open('results.txt', 'w') as f:
    #save latex_string to txt
    f.write(latex_string)
