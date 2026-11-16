import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=600)
ax.set_xlim(0, 146); ax.set_ylim(0, 118); ax.axis('off')
C = dict(rule='#E8EEF7', sup='#E6F2EA', ano='#FBEFE3', fuse='#EDE7F6', io='#F2F2F2', edge='#333333')


def box(x, y, w, h, txt, fc, bold=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec=C['edge'], lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.2)


def arr(x1, y1, x2, y2, txt=None, dx=1.5, ha='left'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.7, color=C['edge']))
    if txt:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, txt, fontsize=6.5, ha=ha, va='center', style='italic')


box(30, 108, 60, 7, 'Pre-processed network flow $\\mathbf{x}$', C['io'])
arr(60, 108, 60, 102.5)
box(10, 91, 100, 11, 'Tier 1: induced signature rules $\\mathcal{R}$\n(attack-pure tree leaves with no benign match in $\\mathcal{D}_{cal1}$)', C['rule'])
ax.plot([110.5, 128, 128], [96.5, 96.5, 47.5], color=C['edge'], lw=0.7); arr(128, 50, 128, 46.6)
ax.text(129.5, 80, 'match:\n$G(\\mathbf{x})=-1-u(\\mathbf{x})$', fontsize=6.5, style='italic', ha='left', va='center')
arr(60, 91, 60, 85, 'no match')
ax.plot([31, 89], [85, 85], color=C['edge'], lw=0.7)
arr(31, 85, 31, 80.5); arr(89, 85, 89, 80.5)
box(4, 68, 54, 12, 'Tier 2: supervised LightGBM\nscore $s_s(\\mathbf{x})$', C['sup'])
box(62, 68, 54, 12, 'Tier 3: benign-only\nautoencoder, score $s_a(\\mathbf{x})$', C['ano'])
arr(31, 68, 31, 63.5); arr(89, 68, 89, 63.5)
box(4, 52, 54, 11, 'conformal p-value $p_s(\\mathbf{x})$\nagainst benign flows of $\\mathcal{D}_{cal1}$', C['sup'], fs=7)
box(62, 52, 54, 11, 'conformal p-value $p_a(\\mathbf{x})$\nagainst benign flows of $\\mathcal{D}_{cal1}$', C['ano'], fs=7)
arr(31, 52, 50, 46.5); arr(89, 52, 72, 46.5)
box(4, 32, 132, 14, 'Fusion $T_\\lambda=\\min\\{p_s/(1-\\lambda),\\,p_a/\\lambda\\}$ with $\\lambda=\\lambda^*$ gives the system statistic $G(\\mathbf{x})$.\nAlert iff the conformal p-value of $G(\\mathbf{x})$ against the benign flows of $\\mathcal{D}_{cal2}$ is at most $\\alpha$.', C['fuse'], fs=6.4)
arr(45, 32, 45, 25, 'p-value $>\\alpha$')
arr(95, 32, 95, 25, 'p-value $\\leq\\alpha$')
box(28, 18, 34, 7, 'BENIGN', C['io'], bold=True)
box(78, 18, 34, 7, 'ALERT', C['io'], bold=True)
ax.plot([4, 142], [13.5, 13.5], color='#999999', lw=0.5, ls='--')
ax.text(73, 10.5, 'Data roles (all taken from the training partition)', fontsize=7, ha='center', weight='bold')
ax.text(73, 4.5, '$\\mathcal{D}_{fit}$: rule tree, LightGBM, autoencoder.    $\\mathcal{D}_{cal1}$: rule screening, reference scores, budget split $\\lambda^*$\n'
        '(inner leave-one-family-out search).    $\\mathcal{D}_{cal2}$: used once, for the final threshold (Theorem 1).',
        fontsize=6.5, ha='center', va='center', linespacing=1.3)
plt.savefig(ROOT + '/figs/fig1_arch.png', bbox_inches='tight', dpi=600)
