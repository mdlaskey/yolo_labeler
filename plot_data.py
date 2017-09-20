import cPickle as pickle 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "/media/autolab/1tb/data/hsr_clutter_rcnn/output/"
date = "07_13_13_55_16"
noisy_date = "07_14_15_27_17"
file = path + date + "losses.p"
noisy_file = path + noisy_date + "losses.p"
data = pickle.load(open(file, "r"))
noisy_data = pickle.load(open(noisy_file, "r"))

#only saved test on normal run
test = data["test"]

noisy_test = noisy_data["test"]
noisy_train = noisy_data["train"]

def savePlot(name, data, i):
	fig = plt.figure(i)
	ax = fig.add_subplot(i * 100 + i * 10 + i)
	for datapoint in data:
		ax.plot(datapoint[0], label=datapoint[1])
	lgd = ax.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0.)

	fig.savefig(name + ".png", bbox_extra_artists=(lgd,), bbox_inches="tight")


for i, val in enumerate([[[test, "test"]], [[noisy_test, "noisy_test"]], [[noisy_train, "noisy_train"]]]):
	savePlot(val[0][1], val, i + 1)

shortened_test = test[:len(noisy_test)]

savePlot("comparison", [[shortened_test, "test"], [noisy_test, "noisy_test"]], 5)