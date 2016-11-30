import matplotlib.pyplot as plt

def plotFile(filename, plot_title):
    xpoints = []
    ypoints = []
    ypointsExpected = []
    ypointsErrorPlus = []
    ypointsErrorMinus = []
    with open(filename, 'r') as f:
        f.readline()   # skip the first line
        expected = float(f.readline().split()[1])

        for line in f:
            data = line.split()
            xpoints.append(int(data[0]))
            ypoints.append(float(data[1]))
            ypointsExpected.append(expected)
            ypointsErrorPlus.append(expected * 1.005)
            ypointsErrorMinus.append(expected * 0.995)

    plt.plot(xpoints, ypointsExpected, 'g', label='Exact')
    plt.plot(xpoints, ypoints, 'b', label='Estimate')
    plt.plot(xpoints, ypointsErrorPlus, 'r')
    plt.plot(xpoints, ypointsErrorMinus, 'r', label='0.5% Error Bounds')
    plt.ylabel('Volume')
    plt.xlabel('Discretization')
    plt.title(plot_title)
    plt.legend(loc='lower right')

    plt.show()


plotFile("output1k_new.txt", "Approximated Volume on Cylindrical Point Cloud 22000 points")

