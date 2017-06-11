import sys

fo = open('/Users/Sravanthi/Downloads/ML project/es-en/europarl-v7.es-en.es','r')
lines = fo.readlines()
fo.close()

fi = open('test.es','w')
for i in range(25000):
	fi.write(lines[i])
fi.close()


fo = open('/Users/Sravanthi/Downloads/ML project/es-en/europarl-v7.es-en.en','r')
lines = fo.readlines()
fo.close()

fi = open('test.en','w')
for i in range(25000):
	fi.write(lines[i])
fi.close()
