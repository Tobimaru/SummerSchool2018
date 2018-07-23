/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: A program to try MPI_Comm_size and MPI_Comm_rank.   *
 *                                                              *
 * Contents: C-Source                                           *
 ****************************************************************/

#include <stdio.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
    /* declare any variables you need */

    MPI_Init(&argc, &argv);

    /* Get the rank of each process */
    int my_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Get the size of the communicator */
    int total_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &total_size);

    /* Write code such that every process writes its rank and the size of the communicator,
     * but only process 0 prints "hello world*/
    if (my_rank == 0)
    {
        printf("rank: %i, size: %i : hello world\n", my_rank, total_size);
    }
    else
    { 
        printf("rank: %i, size: %i \n", my_rank, total_size);
    }   

    MPI_Finalize();
    return 0;
}
