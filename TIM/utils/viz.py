"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""

import matplotlib.pyplot as plt
import numpy as np


class Ax3DPose(object):
    def __init__(self, ax, label=['GT', 'Pred']):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # 0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27

        # Start and endpoints of our representation (i.e 2 punti per ogni membro, poichè sono rette!)
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1    # indici joint gestiti
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1   # indici joint gestiti
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, c='black', label=label[0])) # caso GT
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, c='black'))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='blue', label=label[1])) # caso predizione
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='blue'))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.legend(loc='lower left')
        self.ax.view_init(120, -90)

    # Questo è il metodo che riduce la dimensione della posa plottata a video!!!
    def update(self, gt_channels, pred_channels, ax):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        #region GT

        assert gt_channels.size == 96, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "black"
        rcolor = "black"

        # MODIFICA: aggiunta label al nodo master nell'origine degli assi
        ax.text(0, 0, 0, 0, color='red')

        for i in np.arange(len(self.I)):
            # disegno memebri come rette passanti per 2 punti x,y,z
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])

            # salvataggio coordinate nei relativi array associati agli assi
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)

            # impostazione colore linea e trasparenza (???)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            self.plots[i][0].set_alpha(0.4)

            # MODIFICA: aggiunte etichette joint slave
            ax.text(x[1], y[1], z[1], str(i + 1), color='red')

        # disegno joint. Gli indici qui riportati sono associati ai 17J con cui viene puntata la matrice
        # derivata dal tensore di posa!
        for index in [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]: # lista movable joints
            joint = gt_vals[index]
            self.ax.scatter(joint[0], joint[1], joint[2], c='black', zorder=2, s=5, alpha=0.5)

        #endregion

        #region predizioni
            
        assert pred_channels.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (32, -1))
        lcolor = "blue"
        rcolor = "blue"

        # disegno rette
        for i in np.arange(len(self.I)):
            # impostazione dei 2 punti rispetto ogni asse per definire il singolo membro
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])

            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)

            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            self.plots_pred[i][0].set_alpha(0.7)

        # disegno joint. Gli indici qui riportati sono associati ai 17J con cui viene puntata la matrice
        # derivata dal tensore di posa!
        for index in [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]: # lista movable joints
            joint = pred_vals[index]
            self.ax.scatter(joint[0], joint[1], joint[2], c='black', zorder=2, s=5, alpha=0.8)

        #endregion

        # impostazione limiti spazio di lavoro rappresentato (root -> valore minimo, r -> valore massimo)
        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

def plot_predictions(xyz_gt, xyz_pred, fig, ax, f_title):   
    # Load all the data
    nframes_pred = xyz_pred.shape[0] # 20x96, con 20 = frame; 96x3 = x,y,z dei 32 joint

    # === Plot and animate ===
    for i in range(nframes_pred):
        ob = Ax3DPose(ax)
        ob.update(xyz_gt[i, :], xyz_pred[i, :], ax)
        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")

        plt.show(block=False) # spazio 3D
        fig.canvas.draw() # disegno posa su spazio    
        plt.pause(0.01) # ritardo di 1ms tra un frame e l'altro durante visualizzazione?

        ax.cla() # pulizia assi ("clear")

        