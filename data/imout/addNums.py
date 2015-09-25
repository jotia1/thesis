f = open('test.txt', 'rU')
o = open('out.txt', 'w')

for i, line in enumerate(f):
	o.write(line.strip())
	o.write(' ')
	o.write(str(i%2))
	o.write('\n')

f.close()
o.close()
