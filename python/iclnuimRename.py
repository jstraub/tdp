import subprocess as subp

for i in range(1600):
  print "mv {}.png {:06}.png".format(i,i)
  subp.call("mv {}.png {:06}.png".format(i,i), shell=True)

