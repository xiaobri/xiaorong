"""Draw per-seed changes and their means from the locked raw replay table."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

LABELS = {
    'moment_plain':'区域统计校正头（普通 focal）',
    'moment_recipe':'区域统计校正头＋原加权排序',
    'weight':'采集组风险加权',
    'rank':'批内困难负例排序',
    'peak':'峰形校正头（普通 focal）',
    'regional_time':'五脑区时序重加权',
    'time':'时序重加权＋全局残差',
    'pairs':'历史配对校正（采集时分数）',
    'pairs_refresh':'历史配对校正＋双方刷新',
    'aux_only':'冻结脑区上的辅助损失',
    'peak_native':'峰形校正头＋原加权排序',
    'time_native':'时序与全局残差＋原加权排序',
    'time_aux':'时序与全局残差＋原加权排序＋区域辅助',
    'fresh_both':'完整模型（参照）',
}


def plot(source, destination):
    font = font_manager.FontProperties(family='Noto Sans CJK JP')
    font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family':font.get_name(), 'font.size':10,'axes.unicode_minus':False})
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    deltas = {key:{int(r['seed']):float(r['delta_vs_baseline']) for r in rows if r['arm']==key} for key in LABELS}
    assert all(set(v)=={0,1,2,3,42} for v in deltas.values())
    av = {a:np.mean(list(d.values())) for a,d in deltas.items()}
    order = sorted((k for k in LABELS if k!='fresh_both'),key=av.get,reverse=True)+['fresh_both']
    fig,ax = plt.subplots(figsize=(9,6.5))
    fig.subplots_adjust(left=.33,right=.86,top=.86,bottom=.12)
    ax.axvline(0,color='#697b80',lw=1)
    low = min(0,min(min(d.values()) for d in deltas.values()))-.15
    high = max(max(d.values()) for d in deltas.values())+.15
    ax.set_xlim(low,high)
    for i,key in enumerate(order):
        d=deltas[key]
        ax.hlines(i,min(d.values()),max(d.values()),lw=.7,color='#adbdc3',zorder=1)
        for seed,jitter in zip((0,1,2,3,42),(-.17,0,.17,-.08,.08)):
            ax.scatter(d[seed],i+jitter,s=24,facecolor='white' if seed in (3,42) else '#448ba6',
                       edgecolor='#266277',lw=.9,zorder=3)
        ax.scatter(av[key],i,marker='D',s=34,color='#182b33',zorder=4)
        ax.text(1.02,i,f'{av[key]:+.2f}',transform=ax.get_yaxis_transform(),va='center',fontsize=10)
    ax.set_yticks(range(len(order)),[LABELS[k] for k in order])
    ax.set_ylim(len(order)-.45,-.65)
    ax.axhline(len(order)-1.5,color='#adb7bd',lw=.7,ls='--')
    ax.set_axisbelow(True)
    ax.grid(axis='x',color='#e5e9eb',lw=.7)
    for spine in ('left','top','right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=7)
    ax.set_xlabel('相对同种子 baseline 的 F1 变化（百分点）')
    ax.set_title('协和验证集上各组件的独立增益',loc='left',fontsize=15,pad=32)
    artists=[Line2D([],[],marker='o',ls='',mfc='white',mec='#266277',label='开发种子 3、42'),
             Line2D([],[],marker='o',ls='',mfc='#448ba6',mec='#266277',label='追加种子 0、1、2'),
             Line2D([],[],marker='D',ls='',color='#182b33',label='五种子均值')]
    ax.legend(handles=artists,ncol=3,loc='lower left',bbox_to_anchor=(-.02,1.001),frameon=False,fontsize=9)
    fig.savefig(destination,dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    plot(args.source,args.destination)
