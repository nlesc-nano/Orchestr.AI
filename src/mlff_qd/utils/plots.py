import numpy as np
import matplotlib.pyplot as plt

from mlff_qd.utils.pca import project_pca2
import logging
logger = logging.getLogger(__name__)

def plot_pca(features, labels, title="PCA", filename="pca.png"):
    red, _  = project_pca2(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange","purple","brown","pink","gray"]
    for lbl in np.unique(labels):
        m = (labels==lbl)
        plt.scatter(red[m,0],red[m,1],label=f"grp{lbl}",c=cmap[lbl%len(cmap)],alpha=0.7)
    plt.legend(); plt.title(title)
    plt.savefig(filename, dpi=300); plt.close()

def plot_outliers(features,labels,outliers,title,filename):
    red, _  = project_pca2(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange"]
    for lbl in np.unique(labels):
        m_all = (labels==lbl)
        m_in = m_all & (outliers==1)
        m_out= m_all & (outliers==-1)
        plt.scatter(red[m_in,0],red[m_in,1],c=cmap[lbl % len(cmap)],label=f"{lbl} in")
        plt.scatter(red[m_out,0],red[m_out,1],c=cmap[lbl % len(cmap)],marker='x',s=50,label=f"{lbl} out")
    plt.title(title); plt.legend()
    plt.savefig(filename, dpi=300); plt.close()
    
def plot_energy_and_forces(energies, forces, filename='analysis.png'):
    """Plot energy-per-frame, energy-per-atom, max/avg force with thresholds."""
    num_frames = len(energies)
    frames     = np.arange(num_frames)
    num_atoms  = forces.shape[1]

    energy_per_atom = energies / num_atoms
    mean_epa = np.mean(energy_per_atom)
    std_epa  = np.std(energy_per_atom)
    epa_2p = mean_epa + 2*std_epa
    epa_3p = mean_epa + 3*std_epa
    epa_2m = mean_epa - 2*std_epa
    epa_3m = mean_epa - 3*std_epa
    chem_p = mean_epa + 0.05
    chem_m = mean_epa - 0.05

    fmagn = np.linalg.norm(forces, axis=2)
    maxF = np.max(fmagn, axis=1)
    avgF = np.mean(fmagn, axis=1)
    mean_avgF = np.mean(avgF)
    std_avgF  = np.std(avgF)

    fig, axes = plt.subplots(4,1, figsize=(10,20))
    # Total Energy
    axes[0].plot(frames, energies, 'o-', label='Total E')
    axes[0].set(title='Total Energy per Frame', xlabel='Frame', ylabel='E (eV)')
    axes[0].legend()
    # Energy/atom
    axes[1].plot(frames, energy_per_atom, 'o-', color='purple', label='E/atom')
    for y, lbl in [(mean_epa,'Mean'), (epa_2p,'Mean+2σ'), (epa_3p,'Mean+3σ'),
                   (epa_2m,''), (epa_3m,''), (chem_p,'±0.05 eV/atom'), (chem_m,'')]:
        axes[1].axhline(y, linestyle='--' if 'σ' in lbl else ':', color='gray', label=lbl)
    axes[1].set(title='Energy per Atom', xlabel='Frame', ylabel='E/N (eV)')
    axes[1].legend()
    # Max force
    axes[2].plot(frames, maxF, 'o-', color='red', label='Max F')
    axes[2].set(title='Max Force per Frame', xlabel='Frame', ylabel='|F| (eV/Å)')
    axes[2].legend()
    # Avg force
    axes[3].plot(frames, avgF, 'o-', color='green', label='Avg F')
    axes[3].axhline(mean_avgF, linestyle='--', color='gray', label='Mean')
    axes[3].axhline(mean_avgF+2*std_avgF, linestyle='--', color='orange', label='Mean+2σ')
    axes[3].axhline(mean_avgF+3*std_avgF, linestyle='--', color='red', label='Mean+3σ')
    axes[3].set(title='Average Force per Frame', xlabel='Frame', ylabel='|F| (eV/Å)')
    axes[3].legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    logger.info(f"[Plot] Energy/force plots saved to {filename}")
