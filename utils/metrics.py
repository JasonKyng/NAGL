import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor


# GPU efficient auroc calculation based on adeval package
from adeval.cuda_mem_effic import auroc

# CPU based metrics calculation
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score

class FewShotMetric(object):
    '''
    Image-level and pixel-level metrics for few-shot segmentation
    including I-AUROC, I-F1, I-ACC, P-AUROC, P-F1, P-mIoU
    '''
    def __init__(self, products):
        self.products = products
        self.stat_buffer = {}
        for product in products:
            self.stat_buffer[product] = {}
            self.stat_buffer[product]['i_score'] = []
            self.stat_buffer[product]['i_gt'] = []
            self.stat_buffer[product]['p_score_map'] = []
            self.stat_buffer[product]['p_gt'] = []

    def update(self, i_logits, i_label, p_logits, p_label, products_list):
        for i, product in enumerate(products_list):
            self.stat_buffer[product]['i_score'].append(i_logits[i, 1].item())
            self.stat_buffer[product]['i_gt'].append(i_label[i].item())
            score_map = ((p_logits[i, 1, ...] + 1 - p_logits[i, 0, ...])/2).detach().cpu().numpy()
            self.stat_buffer[product]['p_score_map'].append(gaussian_filter(score_map, 4.0))
            self.stat_buffer[product]['p_gt'].append(p_label[i].detach().cpu().numpy())

    def get_scores(self):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.calculate_scores, product): product for product in self.products}
            for future in futures:
                results.append(future.result())

        self.results = np.array(results)
        mean_results = self.results.mean(0)
        return mean_results[0], mean_results[1] 
    
    def calculate_scores(self, product):
        # image level
        i_score = np.array(self.stat_buffer[product]['i_score'])
        i_gt = np.array(self.stat_buffer[product]['i_gt'])
        i_auroc = auroc(torch.tensor(i_score), torch.tensor(i_gt))
        # i_auroc = roc_auc_score(i_gt, i_score)

        # pixel level
        p_score_map = np.stack(self.stat_buffer[product]['p_score_map'])
        p_gt = np.stack(self.stat_buffer[product]['p_gt'])
        p_auroc = auroc(torch.tensor(p_score_map.flatten()), torch.tensor(p_gt.flatten()))
        # p_auroc = roc_auc_score(p_gt.flatten(), p_score_map.flatten())
        return i_auroc, p_auroc

    def print_metrics(self):
        metrics = self.results
        print(f'{"Product":<20} {"I-AUROC":<20} {"P-AUROC":<20}')
        for i, product in enumerate(self.products):
            print(f'{product:<20} {metrics[i][0]:<20.4f} {metrics[i][1]:<20.4f}')
        metrics_mean = metrics.mean(0)
        print(f'{"Mean":<20} {metrics_mean[0]:<20.4f} {metrics_mean[1]:<20.4f}')
        
    def reset(self):
        self.stat = np.zeros((self.n_class + 1, 3))     # +1 for bg, 3 for tp, fp, fn