## use "icc" to compile source files.
##CC = icc
CC = gcc
## Use the Intel Linker
##LD = icpc
LD = g++

LAPTOP = 1

#INCLUDE_PATH=/opt/intel/mkl/8.0.1/include

#INCLUDE_PATH_VECLIB=/System/Library/Frameworks/vecLib.framework/Versions/A/Headers

#LIB_PATH=/opt/intel/mkl/8.0.1/lib/32

## ifdef WORK
#LIB_PATH=/opt/intel/mkl/8.0.1/lib/em64t/
# LIB_PATH = .
## endif

## Compiler flags go here.
CFLAGS = -O2 -c 
##CFLAGS = -g -c
## Linker flags go here. 
##-L$(LIB_PATH)

## use this command to erase files.
RM = /bin/rm -f

## list of generated object files.


DP_MATCH_OBJS = DP_Shape_Match_SRVF_nDim.o functions.o shape.o
DP_COORD_MATCH_OBJS = DP_Shape_Match_Coord_nDim.o functions.o shape.o

## program executable file name.


DP_MATCH_PROG = DP_Shape_Match_SRVF_nDim
DP_COORD_MATCH_PROG = DP_Shape_Match_Coord_nDim

src_dir = ../src/
bin_dir = ./bin/

## top-level rule, to compile everything.
all:	$(DP_MATCH_PROG) $(DP_COORD_MATCH_PROG)

## rule to link the program

$(DP_MATCH_PROG):$(DP_MATCH_OBJS)
	$(LD) $(DP_MATCH_OBJS) $(LDFLAGS) -o $(DP_MATCH_PROG)

$(DP_COORD_MATCH_PROG):$(DP_COORD_MATCH_OBJS)
	$(LD) $(DP_COORD_MATCH_OBJS) $(LDFLAGS) -o $(DP_COORD_MATCH_PROG)


## rule for file elastic_geodesic.o

	

shape.o: $(src_dir)shape.cpp $(src_dir)shape.h 
	$(CC) $(CFLAGS) $(src_dir)shape.cpp	

functions.o: $(src_dir)functions.cpp $(src_dir)functions.h $(src_dir)shape.h 
	$(CC) $(CFLAGS) $(src_dir)functions.cpp

	
DP_Shape_Match_SRVF_nDim.o: $(src_dir)DP_Shape_Match_SRVF_nDim.cpp
	$(CC) $(CFLAGS) $(src_dir)DP_Shape_Match_SRVF_nDim.cpp

DP_Shape_Match_Coord_nDim.o: $(src_dir)DP_Shape_Match_Coord_nDim.cpp
	$(CC) $(CFLAGS) $(src_dir)DP_Shape_Match_Coord_nDim.cpp


## rule for cleaning re-compilable files.
clean:
	$(RM) $(GEO_PROG) $(GEO_OBJS)	
	$(RM) $(MEAN_PROG) $(MEAN_OBJS)	
	$(RM) $(DP_MATCH_PROG) $(DP_COORD_MATCH_PROG) $(DP_MATCH_OBJS)	$(DP_COORD_MATCH_OBJS)
