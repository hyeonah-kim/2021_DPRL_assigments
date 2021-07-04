import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from GridWorldEnv_DP import GWE_Trap
from DPAgentOnGW import DPAgentOnGW


class Main_GUI(QMainWindow):
    def __init__(self, agent):
        super().__init__()
        uic.loadUi('Main_GUI_DP.ui', self)
        self.agent_call = agent
        self.Combobox_activated()
        self.ChooseEnvironment.activated.connect(self.Combobox_activated)
        self.ChooseAgent.activated.connect(self.Combobox_activated)
        self.QButton.toggled.connect(self.QVButton_activated)
        self.steps_run.clicked.connect(self.StepButton_Clicked)
        self.ClearAllButton.clicked.connect(self.ClearAll_Clicked)
        self.area = DrawGraphs(self)
        self.PaintArea.addWidget(self.area)
        self.upLayout.addWidget(self.PaintArea)
        self.make_environment()
        self.show()

    def get_env_index(self):
        return self.ChooseEnvironment.currentIndex()

    def get_agent_index(self):
        return self.ChooseAgent.currentIndex()

    def make_environment(self):
        selectedEnv = self.get_env_index()

        self.gw = GWE_Trap()
        self.update_envs_custom()
        self.lowV.setText('-6')
        self.highV.setText('-2')

        self.prefix_example()
        self.DiscountRate.setText(str(self.gw.gamma))
        self.area.set_env(selectedEnv)
        return self.gw

    def get_agent(self):
        selectedAgent = self.get_agent_index()
        if selectedAgent <= 2:
            self.agent = self.dp_agent
        return self.agent

    def Combobox_activated(self):
        selected = self.get_env_index() * self.get_agent_index()
        if selected < 1:
            self.tab.setEnabled(False)
        else:
            self.tab.setEnabled(True)
            self.PaintArea.setFrameStyle(QFrame.Panel)
            self.area.vision_on()
            self.make_environment()

    def prefix_example(self):
        self.dp_agent = self.agent_call(self.gw)
        self.agent = self.get_agent()
        self.nc.setText(str(self.gw.nc))
        self.nr.setText(str(self.gw.nr))
        self.set_enable(False)
        self.area.initializeQValue()
        self.area.set_nc(self.gw.nc)
        self.area.set_nr(self.gw.nr)
        self.area.update()

    def update_envs_custom(self):
        if self.get_env_index() == 1:  # custom
            self.gw.ec = int(self.ec.text())
            self.gw.er = int(self.er.text())
            self.gw.step_penalty = float(self.StepPenalty.text())
            self.gw.reward = float(self.Reward.text())
            self.gw.nr = int(self.nr.text())
            self.gw.nc = int(self.nc.text())
            self.area.set_ex(self.gw.ec)
            self.area.set_ey(self.gw.er)
        self.area.set_nc(self.gw.nc)
        self.area.set_nr(self.gw.nr)

        # reconnect env to agent
        self.dp_agent = self.agent_call(self.gw)
        self.agent = self.get_agent()
        self.agent.gamma = float(self.DiscountRate.text())

    def set_enable(self, Bool):
        self.nr.setEnabled(Bool)
        self.nc.setEnabled(Bool)
        self.er.setEnabled(Bool)
        self.ec.setEnabled(Bool)
        self.StepPenalty.setEnabled(Bool)
        self.DiscountRate.setEnabled(Bool)
        self.Reward.setEnabled(Bool)

    # update environment (nc)
    def nc_edit_activated(self):
        self.area.set_nc(self.nc.text())

    # update environment (nr)
    def nr_edit_activated(self):
        self.area.set_nr(self.nr.text())

    # select (Q or V)
    def QVButton_activated(self):
        if self.QButton.isChecked():
            self.area.chooseQorV(0)
        elif self.VButton.isChecked():
            self.area.chooseQorV(1)

    # (Important) Step Button is clicked
    def StepButton_Clicked(self):
        n_out = int(self.n_out.text())
        n_in = int(self.n_in.text())
        self.agent = self.get_agent()

        self.agent.gamma = float(self.DiscountRate.text())

        selectedAgent = self.get_agent_index()
        if selectedAgent == 1:
            self.agent.policy_iteration(n_out, n_in)
        elif (selectedAgent == 2):
            self.agent.value_iteration(n_out)
        # self.statusBar.showMessage(self.agent.getStatusMsg())
        self.area.update()

    # update environment Ending X
    def ec_changed(self):
        self.area.set_ex(self.ec.text())

    # update environment Ending Y
    def er_changed(self):
        self.area.set_ey(self.er.text())

    # Clear all the settings and QValue
    def ClearAll_Clicked(self):
        self.agent.reset_all()
        self.area.update()
        self.update_envs_custom()


class DrawGraphs(QWidget):
    def __init__(self, main_gui):
        super(DrawGraphs, self).__init__()
        self.setPalette(QPalette(Qt.white))
        self.setAutoFillBackground(True)
        self.vision = 0
        self.nc = 5
        self.nr = 5
        self.plot_type = 0  # 0 is Q and 1 is V
        self.initializeQValue()
        self.sx = 0
        self.sy = 0
        self.ex = 3
        self.ey = 1
        self.envs = 1
        self.main = main_gui

    # At start, visualization on
    def vision_on(self):
        self.vision = int(1)
        self.update()

    # updated nc
    def set_nc(self, nc):
        if int(nc) > 0:
            self.nc = int(nc)
        self.update()

    # updated nr
    def set_nr(self, nr):
        if int(nr) > 0:
            self.nr = int(nr)
        self.update()

    # updated sx
    def set_sx(self, sx):
        sx = int(sx)
        if (sx >= 0 and sx < self.nc and (sx != self.ex or self.sy != self.ey)):
            self.sx = sx
        self.update()

    # updated sy
    def set_sy(self, sy):
        if (int(sy) >= 0 and int(sy) < self.nr and (int(sy) != self.ey or self.sx != self.ex)):
            self.sy = int(sy)
        self.update()

    # updated ex
    def set_ex(self, ex):
        if(int(ex) >= 0 and int(ex) < self.nc and (int(ex) != self.sx or self.sy != self.ey)):
            self.ex = int(ex)
        self.update()

    # updated ey
    def set_ey(self, ey):
        if(int(ey) >= 0 and int(ey) < self.nr and (int(ey) != self.sy or self.sx != self.ex)):
            self.ey = int(ey)
        self.update()

    # updated environment
    def set_env(self, env):
        self.envs = env
        self.update()

    # choose Q or V
    def chooseQorV(self, chosen):
        self.plot_type = int(chosen)
        self.update()

    # initialize Q Value
    def initializeQValue(self):
        self.QValue = np.zeros((self.nr, self.nc, 4))
        self.VValue = np.zeros((self.nr, self.nc))

    # Not used yet. So it's incomplete. Setting Q Value Function
    def setQV(self):
        self.QValue = self.main.agent.get_Q_2D()
        self.VValue = self.main.agent.get_V_2D()

    def get_color(self, VQ):
        low = float(self.main.lowV.text())
        high = float(self.main.highV.text())
        max_min = high - low
        c = VQ
        c = c * (c >= low) + low * (c < low)
        c = c * (c <= high) + high * (c > high)
        c = np.floor(30 + 225 / (max_min+0.0001) * (c - low))
        return c

    # Drawing Part - override(paintEvent), and the optimal action arrow is not added yet.
    def paintEvent(self, QPaintEvent):
        self.setQV()
        VfromQ = (self.VValue.min() == 0.0) and (self.VValue.max() == 0.0)

        if (self.plot_type == 0) or VfromQ:
            C = self.get_color(self.QValue)
        else:
            C = self.get_color(self.VValue)
        if self.vision == 1:
            p = QPainter(self)
            self.x_size = self.width() / self.nc
            self.y_size = self.height() / self.nr
            xoff = self.x_size / 4
            yoff = self.y_size / 4
            for x in range(self.nc):
                for y in range(self.nr):
                    ptx = self.x_size * x
                    pty = self.y_size * y
                    ptxa = self.x_size * (x + 1)
                    ptya = self.y_size * (y + 1)
                    ptxc = self.x_size * (x + 0.5)
                    ptyc = self.y_size * (y + 0.5)
                    ptyc5 = ptyc + 5
                    if self.plot_type == 0:  # Plot Q Value
                        for a in range(0, 4, 1):
                            q = round(self.QValue[y, x, a], 2)
                            color = C[y, x, a]
                            p.setBrush(QColor(255-color, 255-color/2, 255-color))
                            if a == 2:  # Up
                                p.drawPolygon(QPointF(ptx, pty), QPointF(ptxa, pty), QPointF(ptxc, ptyc))
                                p.drawText(ptxc - xoff*0.6, ptyc5 - yoff, str(q))
                            elif a == 3:  # Down
                                p.drawPolygon(QPointF(ptx, ptya), QPointF(ptxa, ptya), QPointF(ptxc, ptyc))
                                p.drawText(ptxc - xoff*0.6, ptyc5 + yoff, str(q))
                            elif a == 1:  # Right
                                p.drawPolygon(QPointF(ptxa, ptya), QPointF(ptxa, pty), QPointF(ptxc, ptyc))
                                p.drawText(ptxc + xoff*0.6, ptyc5, str(q))
                            elif a == 0:  # left
                                p.drawPolygon(QPointF(ptx, pty), QPointF(ptx, ptya), QPointF(ptxc, ptyc))
                                p.drawText(ptxc - xoff*1.8, ptyc5, str(q))
                    else:
                        if VfromQ:
                            v = round(np.max(self.QValue[y, x, :]), 2)
                            color = np.max(C[y, x, :])
                        else:
                            v = round(self.VValue[y, x], 2)
                            color = C[y, x]
                        p.setBrush(QColor(255 - color, 255 - color / 2, 255 - color))
                        p.drawPolygon(QPointF(ptx, pty), QPointF(ptxa, pty), QPointF(ptxa, ptya), QPointF(ptx, ptya))
                        p.drawText(ptxc-15, ptyc-15, str(v))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main_GUI(DPAgentOnGW)
    sys.exit(app.exec_())
