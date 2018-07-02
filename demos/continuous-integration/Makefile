COMMONDIR=.

CC=mpicc
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng

EXE=heat_mpi
OBJS=core.o setup.o utilities.o io.o main.o
OBJS_TEST=core.o setup.o utilities.o io.o test.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

TEST=test.c

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
core.o: core.c heat.h
utilities.o: utilities.c heat.h
setup.o: setup.c heat.h
io.o: io.c heat.h
main.o: main.c heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CC)
$(OBJS_TEST): C_COMPILER := $(CC)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CC) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

test: $(OBJS_TEST) $(OBJS_PNG)
	$(CC) $(CCFLAGS) $(OBJS_TEST) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS) -lcunit

%.o: %.c
	$(C_COMPILER) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) test a.out *.o *.png *~
