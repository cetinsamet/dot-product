/*
 *   dotProduct.c
 *
 *   An MPI program that computes the inner product of two vectors
 *
 *   Written by cetinsamet -*- cetin.samet@metu.edu.tr
 *   April, 2019
 *
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>


int main(int argc, char *argv[]) {

    int i;
    int rank, size; 

    double minStart, maxEnd;
    double dotP, locDotP;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ORDER	= atoi(argv[1]);
    int locSize = ORDER / size;

    double* locU;	locU = (double*) malloc(locSize * sizeof(double));
    double* locV;	locV = (double*) malloc(locSize * sizeof(double));
    
    for (i = 0; i < locSize; ++i) {
        locU[i] = i;
        locV[i] = i;
    }
    
    double pStart = MPI_Wtime();    // <-- Start Time
    locDotP = cblas_ddot(locSize, locU, 1, locV, 1);
    MPI_Reduce(&locDotP, &dotP, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double pEnd = MPI_Wtime();      // <-- Stop Time
    
    MPI_Reduce(&pStart,  &minStart, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pEnd, &maxEnd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    	printf("Dot product: %f\tVector size: %d\tElapsed time: %f\n", dotP, ORDER, maxEnd - minStart);

    MPI_Finalize();
    return 0;
}
