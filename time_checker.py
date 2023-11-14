import time
from IPython.display import display
from inspect import cleandoc


class printer(str):
	def __init__(self, val : str) -> None:
		super().__init__()
		self = cleandoc(val)
	
	def __repr__(self):
		return self


class Time_checker():
	def __init__(self, nb_check) -> None:
		self.sums = [0]*nb_check
		self.current = -1
		self.nb_it = 1
		self.last_time = 0
		self.nb_check = nb_check
		self.dh = display(repr(self), display_id= True)
		self.last_update = 0

	
	def __repr__(self) -> str:
		repr_obj = ""
		for i,sum in enumerate(self.sums):
			repr_obj += f" moyenne {i} : {sum/self.nb_it}"
		return printer(repr_obj)

	def check(self):
		t = time.time()
		if self.current >= 0:
			self.sums[self.current] += t - self.last_time
		self.last_time = t
		self.current += 1

	def update(self):
		if self.current != self.nb_check:
			raise ValueError("you did not check enough value this turn")
		self.current = -1
		if time.time() - self.last_update > 1:
			self.last_update = time.time()
			self.dh.update(repr(self))
		self.nb_it += 1
	
	def check_update(self):
		self.check()
		self.update()

	def end_update(self):
		self.dh.update(repr(self))