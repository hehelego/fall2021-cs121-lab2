CC=nvcc
flag=
debug=0
define=-D DEBUG=$(debug)
libs=-lcurand
src=src/main.cu src/cuckoo.cu src/common.cu
binary=bin/cuckoo

all:
	$(CC) \
		$(flag) \
		$(define) \
		$(libs) \
		$(src) -o $(binary) \

clean:
	rm bin/* -f
