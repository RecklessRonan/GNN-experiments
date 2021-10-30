#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:36:40 2021

@author: sitaoluan
"""
import matplotlib.pyplot as plt  
import numpy as np
def plot_results(dataset_name, GCN, snowball_2, snowball_3,
                 ACM_GCN, ACMII_GCN, ACM_snowball_2, ACMII_snowball_2,
                 ACM_snowball_3, ACMII_snowball_3, SOTA, SOTA_name, with_legend = True):
    
    for i, dataset in zip(range(len(dataset_name)), dataset_name):
        # x_label = ["ACM_GCN", "ACMII_GCN", "ACM_snowball_2", "ACMII_snowball_2", "ACM_snowball_3", "ACMII_snowball_3"]
        x_label = ["GCN", "snowball-2", "snowball-3"]
        
        
        # Setting the positions and width for the bars
        pos = [0, 0.38, 0.83]
        width = 0.1 
        plt.rcParams.update({'font.size': 28})
        # Plotting the bars
        fig, ax = plt.subplots(figsize=(8,6))
        plt.plot([-0.05,1.05], [SOTA[0][i],SOTA[0][i]], color='m',linestyle='--')
        l1 = plt.fill_between([-0.05,1.05], np.clip(SOTA[0][i] - SOTA[1][i], 0, 100), np.clip(SOTA[0][i] + SOTA[1][i], 0, 100), color='m', alpha = 0.2)
        
        l2 = plt.bar(0, GCN[0][i], width,
                         alpha=0.5,
                         color='r',
                         yerr= GCN[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        l3 = plt.bar(0+ 1*width, ACM_GCN[0][i], width,
                         alpha=0.5,
                         color='g',
                         yerr= ACM_GCN[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        l4 = plt.bar(0+ 2*width, ACMII_GCN[0][i], width,
                         alpha=0.5,
                         color='b',
                         yerr= ACMII_GCN[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        
        plt.bar(0.4, snowball_2[0][i], width,
                         alpha=0.5,
                         color='r',
                         yerr= snowball_2[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        
        plt.bar(0.4+ 1*width, ACM_snowball_2[0][i], width,
                         alpha=0.5,
                         color='g',
                         yerr= ACM_snowball_2[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        
        
        plt.bar(0.4+ 2*width, ACMII_snowball_2[0][i], width,
                         alpha=0.5,
                         color='b',
                         yerr= ACMII_snowball_2[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        plt.bar(0.8, snowball_3[0][i], width,
                         alpha=0.5,
                         color='r',
                         yerr= snowball_3[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        plt.bar(0.8+ 1*width, ACM_snowball_3[0][i], width,
                         alpha=0.5,
                         color='g',
                         yerr= ACM_snowball_3[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        plt.bar(0.8+ 2*width, ACMII_snowball_3[0][i], width,
                         alpha=0.5,
                         color='b',
                         yerr= ACMII_snowball_3[1][i],
                         #label=Label_L4L5_DP[3]
                         )
        
        # plt.plot([-0.05,1.05], [SOTA[0][i],SOTA[0][i]], color='m')
        # plt.axhline(y=SOTA[0][i], color='m', linestyle='--')
        # ax.axhspan(-0.05, 1.05, np.clip(SOTA[0][i] - SOTA[1][i], 0, 100), np.clip(SOTA[0][i] + SOTA[1][i], 0, 100), alpha=0.1, color='m')
        # plt.plot([-0.05,1.05], [SOTA[0][i],SOTA[0][i]], color='m',linestyle='--')
        # plt.fill_between([-0.05,1.05], np.clip(SOTA[0][i] - SOTA[1][i], 0, 100), np.clip(SOTA[0][i] + SOTA[1][i], 0, 100), color='m', alpha = 0.3)
        # # plt.plot([p + 3*width for p in pos], L4L5DP_Takahashi, width,
                         # alpha=0.5,
                         # color='k',
                         #yerr= 0,
                         #label=Label_L4L5_DP[2]
                         # )
        
        #    plt.errorbar(Sato_Mean_L4L5_Upr,Sato_Mean_L4L5_Upr, xerr = offsets_Sato_L4L5_Upr, fmt='b')
        #    plt.errorbar([p + 3*width for p in pos], Sato_Mean_L4L5_F53, xerr = offsets_Sato_L4L5_F53, fmt='b')
        #    plt.errorbar([p + 3*width for p in pos], Sato_Mean_L4L5_E30, xerr = offsets_Sato_L4L5_E30, fmt='b')
        
        # Setting axis labels and ticks
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(dataset + ' (SOTA='+ SOTA_name[i]+ ")")
        ax.set_xticks([p + 1 * width for p in pos])
        ax.set_xticklabels(x_label)
        
        # Setting the x-axis and y-axis limits
        # plt.xlim(min(pos)-width, max(pos)+width*4)
        plt.ylim([min(GCN[0][i], snowball_2[0][i],snowball_3[0][i])-10, 
                  max(ACM_GCN[0][i],ACMII_GCN[0][i],ACM_snowball_2[0][i],ACMII_snowball_2[0][i],ACM_snowball_3[0][i],ACMII_snowball_3[0][i])+3])
        
        # Adding the legend and showing the plot
        plt.legend([l1,l2,l3,l4],["SOTA", 'baseline GNNs', 'ACM-baseline', 'ACMII-baseline'],bbox_to_anchor=(1.7, 1), loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.grid() +"("+ SOTA_name[i] +")"

        if not with_legend:
            ax.get_legend().remove()
            # ax[1].get_legend().remove()
            fig_name = f"./plots/fig_{dataset}.pdf"
        else:
            fig_name = f"./plots/fig_{dataset}_legend.pdf"
        # plt.tight_layout()
        
        plt.savefig(fig_name,bbox_inches = 'tight')
        plt.show()
        
if __name__ == "__main__":
    dataset_name = ['Cornell','Wisconsin','Texas','Film', 'Chameleon', 'Squirrel']
    GCN = [[82.46,75.50,83.11,35.51,64.18,44.76],[3.11,2.92,3.2,0.99,2.62,1.39]]
    snowball_2 = [[82.62,74.88,83.11,35.97,64.99,47.88],[2.34,3.42,3.2,0.66,2.39,1.23]]
    snowball_3 = [[82.95,69.50,83.11,36.00,65.49,48.25],[2.1,5.01,3.2,1.36,1.64,0.94]]
    ACM_GCN = [[94.75,95.75,94.92,41.62,69.04,58.02],[3.8,2.03,2.88,1.15,1.74,1.86]]
    ACMII_GCN = [[95.90,96.62,95.08,41.84,68.38,54.53],[1.83,2.44,2.07,1.15,1.36,2.09]]
    ACM_snowball_2 = [[95.08,96.38,95.74,41.40,68.51,55.97],[3.11,2.59,2.22,1.23,1.7,2.03]]
    ACMII_snowball_2 = [[95.25,96.63,95.25,41.10,67.83,53.48],[1.55,2.24,1.55,0.75,2.63,0.6]]
    ACM_snowball_3 = [[94.26,96.62,94.75,41.27,68.40,55.73],[2.57,1.86,2.41,0.8,2.05,2.39]]
    ACMII_snowball_3 = [[93.61,97.00,94.75,40.31,67.53,52.31],[2.79,2.63,3.09,1.6,2.83,1.57]]
    SOTA = [[91.8,93.87,92.92,39.3,68.14,53.4],[0.63,3.33,0.61,0.27,1.18,1.9]]
    SOTA_name = ["APPNP","MLP-2","GPRGNN","GPRGNN","GAT+JK","GCN+JK"]
    plot_results(dataset_name, GCN, snowball_2, snowball_3,
                 ACM_GCN, ACMII_GCN, ACM_snowball_2, ACMII_snowball_2,
                 ACM_snowball_3, ACMII_snowball_3, SOTA, SOTA_name, with_legend = True)
    
    
    