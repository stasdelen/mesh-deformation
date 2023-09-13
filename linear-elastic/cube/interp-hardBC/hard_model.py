import torch
from remin import callbacks
from remin.residual import make_loader
from remin.solver import Solver, make_trainer
from remin.solver.residual_loss import EagerLoss
from torch import nn


class HardMesh(nn.Module):
	def __init__(self, distance, Up):
		super().__init__()
		# Distance function
		self.distance = distance
		self.Xp, self.Yp, self.Zp = Up[:,0:1], Up[:,1:2], Up[:,2:3]
		self.linear_tanh_stack = nn.Sequential(
			nn.Linear(3, 50),
			nn.Tanh(),
			nn.Linear(50, 50),
			nn.Tanh(),
			nn.Linear(50, 3)
		)
	
	def forward(self, x, y, z):
		Upar = torch.hstack((x, y, z))
		Uhat = self.linear_tanh_stack(Upar)
		X_nn, Y_nn, Z_nn = Uhat[:,0:1], Uhat[:,1:2], Uhat[:,2:3]
		X = self.Xp + self.distance * X_nn
		Y = self.Yp + self.distance * Y_nn
		Z = self.Zp + self.distance * Z_nn
		return X, Y, Z
	
	def calc(self, Uin):
		Uhat = self.linear_tanh_stack(Uin)
		X_nn, Y_nn, Z_nn = Uhat[:,0:1], Uhat[:,1:2], Uhat[:,2:3]
		return torch.hstack(
			(self.Xp + self.distance * X_nn,
			 self.Yp + self.distance * Y_nn,
			 self.Zp + self.distance * Z_nn)
		)
	
def trainHardModel(residuals, distance, U_p, eval_vertices, name):
	
	hardModel = HardMesh(distance, U_p)

	loader = make_loader(
		residuals,
		fully_loaded=True
	)

	epochs = 1#0000
	lr = 1e-3
	gamma = 0.95

	optimizer = torch.optim.Adam(hardModel.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)
	resloss = EagerLoss(nn.MSELoss())
	trainer = make_trainer(loader,
						   optimizer=optimizer,
						   scheduler=scheduler,
						   residual_loss=resloss)
	
	solver = Solver(hardModel,
					name=name,
					save_folder='./outputs/' + name,
					trainer=trainer)
	
	solver.reset_callbacks(
		callbacks.TotalTimeCallback(),
		callbacks.SaveCallback(),
		callbacks.LogCallback(log_epoch=1000, log_progress=10),
		callbacks.ToleranceCallback(tolerance=5e-11),
		callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)
	return hardModel.calc(eval_vertices).cpu().detach().numpy()